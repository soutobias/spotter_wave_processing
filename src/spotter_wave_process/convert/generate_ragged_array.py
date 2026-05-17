import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
from datetime import datetime, timezone
from spotter_wave_process.convert.variable_mapping import COLUMN_MAP, SPECTRAL_COLUMNS

def parse_semicolon_array(value) -> np.ndarray:
    """
    Parse one cell containing semicolon-separated spectral values.

    Examples
    --------
    "0.0293;0.03906;0.04883" -> np.array([...])
    "-" -> empty array
    NaN -> empty array
    """
    if pd.isna(value):
        return np.array([], dtype=np.float64)

    value = str(value).strip()

    if value in {"", "-", "nan", "NaN", "None"}:
        return np.array([], dtype=np.float64)

    return np.array(
        [float(item) if item not in {"", "-", "nan", "NaN"} else np.nan for item in value.split(";")],
        dtype=np.float64,
    )

def read_spotter_csv(
    path: str | Path,
    trajectory_id: str,
    *,
    drop_missing_spectra: bool = True,
) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=",",
        na_values=["-", ""],
        keep_default_na=True,
    )

    # Keep only columns we know how to handle.
    existing_columns = [col for col in COLUMN_MAP if col in df.columns]
    df = df[existing_columns].rename(columns=COLUMN_MAP)

    # Parse time.
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)

    # Add trajectory metadata.
    df["trajectory"] = trajectory_id
    df["sourceFile"] = str(path)

    # Parse spectral columns.
    parsed_spectral_columns = [
        COLUMN_MAP[col] for col in SPECTRAL_COLUMNS if COLUMN_MAP[col] in df.columns
    ]

    for name in parsed_spectral_columns:
        df[name] = df[name].apply(parse_semicolon_array)

    if drop_missing_spectra:
        # Require at least frequency and varianceDensity to exist.
        df = df[
            (df["frequency"].apply(len) > 0)
            & (df["varianceDensity"].apply(len) > 0)
        ].copy()

    # Convert scalar columns to numeric.
    non_numeric = {
        "time",
        "trajectory",
        "sourceFile",
        "processingSource",
        *parsed_spectral_columns,
    }

    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_spotter_csvs(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frames = []

    for spot_dir in sorted(root.glob("Spot-*")):
        if not spot_dir.is_dir():
            continue

        trajectory_id = spot_dir.name.upper()

        for csv_file in sorted(spot_dir.glob("*/*.csv")):
            df = read_spotter_csv(
                csv_file,
                trajectory_id=trajectory_id,
                drop_missing_spectra=True,
            )

            if not df.empty:
                frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No usable CSV files found below {root}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["trajectory", "time"]).reset_index(drop=True)

    return df

def validate_spectral_lengths(df: pd.DataFrame) -> int:
    spectral_vars = [
        "frequency",
        "frequencyBandwidth",
        "a1",
        "b1",
        "a2",
        "b2",
        "varianceDensity",
        "direction",
        "directionalSpread",
    ]

    lengths = {}

    for var in spectral_vars:
        if var not in df:
            continue

        unique_lengths = sorted(df[var].apply(len).unique())
        lengths[var] = unique_lengths

    bad = {
        var: lens
        for var, lens in lengths.items()
        if len(lens) != 1
    }

    if bad:
        raise ValueError(
            "Inconsistent spectral lengths found:\n"
            + "\n".join(f"{var}: {lens}" for var, lens in bad.items())
        )

    n_frequency = len(df["frequency"].iloc[0])

    for var, lens in lengths.items():
        if lens[0] != n_frequency:
            raise ValueError(
                f"{var!r} has length {lens[0]}, but frequency has length {n_frequency}"
            )

    return n_frequency


def stack_array_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.vstack(df[name].to_numpy()).astype(np.float64)


def dataframe_to_spectral_ragged_dataset(df: pd.DataFrame) -> xr.Dataset:
    df = df.sort_values(["trajectory", "time"]).reset_index(drop=True)

    n_frequency = validate_spectral_lengths(df)

    frequency = np.asarray(df["frequency"].iloc[0], dtype=np.float64)

    trajectories = df["trajectory"].drop_duplicates().to_numpy(dtype=object)

    rowsize = (
        df.groupby("trajectory", sort=False)
        .size()
        .to_numpy(dtype=np.int64)
    )

    scalar_vars = [
        "batteryVoltage",
        "power",
        "humidity",
        "significantWaveHeight",
        "peakPeriod",
        "meanPeriod",
        "peakDirection",
        "peakDirectionalSpread",
        "meanDirection",
        "meanDirectionalSpread",
        "latitude",
        "longitude",
        "windSpeed",
        "windDirection",
        "surfaceTemperature",
        "partition0StartFrequency",
        "partition0EndFrequency",
        "partition0SignificantWaveHeight",
        "partition0MeanPeriod",
        "partition0MeanDirection",
        "partition0MeanDirectionalSpread",
        "partition1StartFrequency",
        "partition1EndFrequency",
        "partition1SignificantWaveHeight",
        "partition1MeanPeriod",
        "partition1MeanDirection",
        "partition1MeanDirectionalSpread",
        "meanBarometricPressure",
    ]

    spectral_vars = [
        "frequencyBandwidth",
        "a1",
        "b1",
        "a2",
        "b2",
        "varianceDensity",
        "direction",
        "directionalSpread",
    ]

    data_vars = {}

    for name in scalar_vars:
        if name in df.columns:
            data_vars[name] = ("index", df[name].to_numpy(dtype=np.float64))

    for name in spectral_vars:
        if name in df.columns:
            data_vars[name] = (
                ("index", "frequency"),
                stack_array_column(df, name),
            )

    data_vars["rowsize"] = ("trajectory", rowsize)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("index", df["time"].to_numpy(dtype="datetime64[ns]")),
            "frequency": ("frequency", frequency),
            "trajectory": ("trajectory", trajectories),
        },
        attrs={
            "title": "Spotter wave buoy spectral parameters and bulk parameters",
            "source": "Spotter wave buoy",
            "featureType": "trajectory",
            "cdm_data_type": "Trajectory",
            "Conventions": "CF-1.8",
            "creation_date": datetime.now(timezone.utc).isoformat(),
        },
    )

    ds["trajectory"].attrs.update(
        {
            "cf_role": "trajectory_id",
            "long_name": "trajectory identifier",
        }
    )

    ds["rowsize"].attrs.update(
        {
            "long_name": "number of observations for this trajectory",
            "sample_dimension": "index",
        }
    )

    ds["time"].attrs.update(
        {
            "standard_name": "time",
            "long_name": "time",
        }
    )

    ds["frequency"].attrs.update(
        {
            "standard_name": "sea_surface_wave_frequency",
            "long_name": "wave frequency",
            "units": "Hz",
        }
    )

    ds["frequencyBandwidth"].attrs.update(
        {
            "long_name": "frequency bandwidth",
            "units": "Hz",
        }
    )

    ds["varianceDensity"].attrs.update(
        {
            "long_name": "wave variance density spectrum",
            "units": "m2 Hz-1",
        }
    )

    for name in ["a1", "b1", "a2", "b2"]:
        ds[name].attrs.update(
            {
                "long_name": f"directional Fourier coefficient {name}",
                "units": "1",
            }
        )

    if "latitude" in ds:
        ds["latitude"].attrs.update(
            {
                "standard_name": "latitude",
                "units": "degrees_north",
            }
        )

    if "longitude" in ds:
        ds["longitude"].attrs.update(
            {
                "standard_name": "longitude",
                "units": "degrees_east",
            }
        )

    if "windSpeed" in ds:
        ds["windSpeed"].attrs.update(
            {
                "standard_name": "wind_speed",
                "units": "m s-1",
            }
        )

    if "meanBarometricPressure" in ds:
        ds["meanBarometricPressure"].attrs.update(
            {
                "long_name": "mean barometric pressure",
                "units": "hPa",
            }
        )

    return ds

def write_spectral_ragged_zarr(
    root: str | Path,
    output_zarr: str | Path,
    *,
    index_chunksize: int = 50_000,
) -> xr.Dataset:
    df = load_all_spotter_csvs(root)
    ds = dataframe_to_spectral_ragged_dataset(df)

    encoding = {}

    for name in ds.data_vars:
        dims = ds[name].dims

        if dims == ("index",):
            encoding[name] = {"chunks": (index_chunksize,)}

        elif dims == ("index", "frequency"):
            encoding[name] = {
                "chunks": (index_chunksize, ds.sizes["frequency"]),
            }

        elif dims == ("trajectory",):
            encoding[name] = {
                "chunks": (ds.sizes["trajectory"],),
            }

    encoding["time"] = {
        "chunks": (index_chunksize,),
        "dtype": "int64",
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "standard",
    }

    ds.to_zarr(
        output_zarr,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    return ds

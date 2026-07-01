"""
Utilities to compare Spotter wave-buoy spectra with ERA5 2D wave spectra.

Main workflow
-------------
The workflow is:

1. Open Spotter and ERA5 Zarr datasets.
2. Decode ERA5 d2fd spectra from log10 spectral density.
3. Integrate ERA5 2D spectra over direction to obtain 1D variance density.
4. Bin Spotter variance density onto the ERA5 frequency grid.
5. Match Spotter observations to nearest ERA5 time and grid point.
6. Build a matchup dataset.
7. Calculate comparison metrics.

Example
-------
from spotter_era5_spectral_comparison import run_workflow

matchup, metrics = run_workflow(
    start_time="2022-01-01",
    end_time="2022-01-07",
    time_tolerance="30min",
    max_distance_m=50_000,
)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


SPOTTER_URL = "https://paidiver-o.s3-ext.jc.rl.ac.uk/waves/spotter_brazil.zarr"
ERA5_URL = "https://paidiver-o.s3-ext.jc.rl.ac.uk/waves/era3/wave_spectra.zarr"

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class MatchupConfig:
    """
    Configuration for Spotter-ERA5 spectral matchup workflow.

    Parameters
    ----------
    spotter_url : str
        URL or path to Spotter Zarr dataset.
    era5_url : str
        URL or path to ERA5 Zarr dataset.
    start_time : str | np.datetime64 | pd.Timestamp | None
        Optional start time used to subset both Spotter and ERA5 before processing.
    end_time : str | np.datetime64 | pd.Timestamp | None
        Optional end time used to subset both Spotter and ERA5 before processing.
    time_tolerance : str
        Maximum allowed time difference for nearest ERA5 match.
    max_distance_m : float
        Maximum allowed distance between Spotter position and ERA5 grid point.
    spotter_chunks : dict | None
        Optional chunks for Spotter dataset.
    era5_chunks : dict | None
        Optional chunks for raw ERA5 dataset.
    era5_spectrum_chunks : dict | None
        Optional chunks for derived ERA5 1D spectrum.
    output_matchup_zarr : str | None
        Optional output Zarr path for matchup dataset.
    output_metrics_netcdf : str | None
        Optional output NetCDF path for metrics dataset.
    """

    spotter_url: str = SPOTTER_URL
    era5_url: str = ERA5_URL

    start_time: str | np.datetime64 | pd.Timestamp | None = None
    end_time: str | np.datetime64 | pd.Timestamp | None = None

    time_tolerance: str = "30min"
    max_distance_m: float = 50_000.0

    spotter_chunks: dict[str, int] | None = None
    era5_chunks: dict[str, int] | None = None
    era5_spectrum_chunks: dict[str, int] | None = None

    output_matchup_zarr: str | None = None
    output_metrics_netcdf: str | None = None


def era5_frequency() -> np.ndarray:
    """
    Return ERA5 wave-model frequency centres in Hz.

    ERA5 uses 30 logarithmically spaced wave frequencies:

        f_n = 0.03453 * 1.1 ** n

    where n = 0, ..., 29.
    """
    return 0.03453 * 1.1 ** np.arange(30)


def era5_direction_degrees() -> np.ndarray:
    """
    Return ERA5 wave-model direction centres in degrees.

    ERA5 uses 24 directions spaced by 15 degrees.
    """
    return np.arange(7.5, 352.5 + 15.0, 15.0)


def frequency_edges_from_centres(
    frequency: np.ndarray,
    *,
    logarithmic: bool = False,
    ratio: float = 1.1,
) -> np.ndarray:
    """
    Calculate frequency-bin edges from frequency-bin centres.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency centres.
    logarithmic : bool
        If True, internal edges are geometric means. This is appropriate
        for ERA5 frequencies. If False, internal edges are arithmetic means.
    ratio : float
        Logarithmic spacing ratio, used only for the first and last edge when
        logarithmic=True.

    Returns
    -------
    np.ndarray
        Frequency-bin edges with length len(frequency) + 1.
    """
    frequency = np.asarray(frequency, dtype=float)

    if frequency.ndim != 1:
        raise ValueError("frequency must be a 1D array.")

    if frequency.size < 2:
        raise ValueError("frequency must contain at least two values.")

    edges = np.zeros(frequency.size + 1, dtype=float)

    if logarithmic:
        edges[1:-1] = np.sqrt(frequency[:-1] * frequency[1:])
        edges[0] = frequency[0] / np.sqrt(ratio)
        edges[-1] = frequency[-1] * np.sqrt(ratio)
    else:
        edges[1:-1] = 0.5 * (frequency[:-1] + frequency[1:])
        edges[0] = frequency[0] - 0.5 * (frequency[1] - frequency[0])
        edges[-1] = frequency[-1] + 0.5 * (frequency[-1] - frequency[-2])

    return edges


def era5_frequency_edges() -> np.ndarray:
    """Return ERA5 frequency-bin edges in Hz."""
    return frequency_edges_from_centres(
        era5_frequency(),
        logarithmic=True,
        ratio=1.1,
    )


def open_datasets(config: MatchupConfig) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Open Spotter and ERA5 Zarr datasets.

    The datasets are opened lazily.
    """
    spotter = xr.open_zarr(config.spotter_url)
    era5 = xr.open_zarr(config.era5_url)

    return spotter, era5


def ensure_chunked(
    obj: xr.Dataset | xr.DataArray,
    chunks: dict[str, int] | None,
) -> xr.Dataset | xr.DataArray:
    """
    Apply Dask chunking if chunks are provided.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset or DataArray to chunk.
    chunks : dict | None
        Chunk mapping. If None, object is returned unchanged.
    """
    if chunks is None:
        return obj

    valid_chunks = {
        dim: size
        for dim, size in chunks.items()
        if dim in obj.dims
    }

    if not valid_chunks:
        return obj

    return obj.chunk(valid_chunks)


def describe_chunks(
    obj: xr.Dataset | xr.DataArray,
    name: str = "object",
) -> None:
    """
    Print Dask chunks for a Dataset or DataArray.
    """
    print(f"\n{name}")

    if isinstance(obj, xr.DataArray):
        print(obj.chunks)
        return

    for var_name, da in obj.data_vars.items():
        if hasattr(da.data, "chunks"):
            print(f"{var_name}: {da.chunks}")

def subset_time_range(
    obj: xr.Dataset | xr.DataArray,
    start_time: str | np.datetime64 | pd.Timestamp | None = None,
    end_time: str | np.datetime64 | pd.Timestamp | None = None,
    *,
    time_name: str = "time",
) -> xr.Dataset | xr.DataArray:
    """
    Subset a Dataset or DataArray by time.

    This works for:
    - time as a dimension coordinate, e.g. ERA5 time(time)
    - time as a non-dimension coordinate, e.g. Spotter time(index)

    For Dask-backed non-dimension time coordinates, it computes only the
    1D boolean indexer, not the full dataset.
    """
    if start_time is None and end_time is None:
        return obj

    if time_name not in obj.coords and time_name not in obj:
        raise KeyError(f"{time_name!r} not found in object.")

    time = obj[time_name]

    # Case 1: time is a proper dimension coordinate, e.g. ERA5 time(time)
    if time_name in obj.dims:
        start = start_time if start_time is not None else time.min().values
        end = end_time if end_time is not None else time.max().values
        return obj.sel({time_name: slice(start, end)})

    # Case 2: time is a coordinate along another dimension, e.g. Spotter time(index)
    if time.ndim != 1:
        raise ValueError(
            f"{time_name!r} is not a dimension and is not 1D. "
            "Cannot infer which dimension to subset."
        )

    index_dim = time.dims[0]

    mask = xr.ones_like(time, dtype=bool)

    if start_time is not None:
        mask = mask & (time >= np.datetime64(start_time))

    if end_time is not None:
        mask = mask & (time <= np.datetime64(end_time))

    # Critical fix:
    # xarray cannot use a lazy Dask boolean mask with drop=True / isel.
    # Compute only this 1D mask.
    if hasattr(mask.data, "compute"):
        mask = mask.compute()

    return obj.isel({index_dim: mask})

def add_era5_spectral_coordinates(era5: xr.Dataset) -> xr.Dataset:
    """
    Add physical ERA5 wave frequency and direction coordinates.
    """
    freq = era5_frequency()
    direction = era5_direction_degrees()

    return era5.assign_coords(
        frequency=("frequencyNumber", freq),
        direction=("directionNumber", direction),
    )


def integrate_era5_to_1d_spectrum(era5: xr.Dataset) -> xr.DataArray:
    """
    Decode ERA5 d2fd and integrate over direction.

    ERA5 d2fd is stored as log10 spectral density. After decoding:

        S = 10 ** d2fd

    The 1D frequency spectrum is:

        E(f) = integral S(f, theta) dtheta

    Returns
    -------
    xr.DataArray
        ERA5 variance density with dimensions originally:
        time, frequencyNumber, values.

        The result is then converted to:
        time, frequency, values.
    """
    era5 = add_era5_spectral_coordinates(era5)

    dtheta = np.deg2rad(15.0)

    spectrum_2d = 10 ** era5["d2fd"]
    spectrum_2d = spectrum_2d.fillna(0.0)

    spectrum_1d = (spectrum_2d * dtheta).sum("directionNumber")
    spectrum_1d = spectrum_1d.rename("varianceDensity_era5")

    spectrum_1d = spectrum_1d.swap_dims({"frequencyNumber": "frequency"})
    spectrum_1d = spectrum_1d.drop_vars("frequencyNumber", errors="ignore")

    return spectrum_1d


def bin_spotter_to_era5_frequency(
    da: xr.DataArray,
    *,
    spot_freq_name: str = "frequency",
    era5_freq: np.ndarray | None = None,
    era5_edges: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Conservatively bin a Spotter spectral variable onto ERA5 frequencies.

    This function is appropriate for variance density. It approximately
    preserves variance by:

    1. multiplying variance density by Spotter df;
    2. summing variance inside each ERA5 frequency bin;
    3. dividing by the ERA5 bin width.

    Parameters
    ----------
    da : xr.DataArray
        Spotter DataArray with a frequency dimension.
    spot_freq_name : str
        Name of the Spotter frequency dimension.
    era5_freq : np.ndarray | None
        ERA5 frequency centres. If None, defaults to ERA5 frequencies.
    era5_edges : np.ndarray | None
        ERA5 frequency edges. If None, defaults to ERA5 edges.

    Returns
    -------
    xr.DataArray
        DataArray binned onto ERA5 frequency centres.
    """
    if spot_freq_name not in da.dims:
        raise ValueError(
            f"{spot_freq_name!r} must be a dimension of the input DataArray."
        )

    if era5_freq is None:
        era5_freq = era5_frequency()

    if era5_edges is None:
        era5_edges = era5_frequency_edges()

    spot_freq = da[spot_freq_name].values
    spot_edges = frequency_edges_from_centres(spot_freq, logarithmic=False)
    spot_df = np.diff(spot_edges)

    out = []

    template = da.isel({spot_freq_name: 0}).drop_vars(
        spot_freq_name,
        errors="ignore",
    )

    for left, right, width in zip(
        era5_edges[:-1],
        era5_edges[1:],
        np.diff(era5_edges),
    ):
        mask = (spot_freq >= left) & (spot_freq < right)

        if not mask.any():
            binned = xr.full_like(template, np.nan)
        else:
            selected_freq = spot_freq[mask]

            weights = xr.DataArray(
                spot_df[mask],
                dims=[spot_freq_name],
                coords={spot_freq_name: selected_freq},
            )

            binned = (
                da.sel({spot_freq_name: selected_freq}) * weights
            ).sum(spot_freq_name) / width

            binned = binned.drop_vars(spot_freq_name, errors="ignore")

        out.append(binned)

    result = xr.concat(
        out,
        dim=xr.IndexVariable(
            spot_freq_name,
            era5_freq,
            attrs={"units": "Hz"},
        ),
    )

    result.name = da.name
    return result


def build_spotter_binned_dataset(
    spotter: xr.Dataset,
    *,
    era5_freq: np.ndarray | None = None,
    era5_edges: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Build a Spotter dataset with variance density binned to ERA5 frequencies.
    """
    if era5_freq is None:
        era5_freq = era5_frequency()

    if era5_edges is None:
        era5_edges = era5_frequency_edges()

    variance_density = bin_spotter_to_era5_frequency(
        spotter["varianceDensity"],
        era5_freq=era5_freq,
        era5_edges=era5_edges,
    ).transpose("index", "frequency")

    binned = xr.Dataset(
        data_vars={
            "varianceDensity": variance_density,
        },
        coords={
            "time": spotter["time"],
            "frequency": variance_density["frequency"],
        },
        attrs={
            **spotter.attrs,
            "frequency_binning": (
                "Spotter spectra conservatively binned to ERA5 frequency bins"
            ),
        },
    )

    copy_vars = [
        "latitude",
        "longitude",
        "significantWaveHeight",
        "peakPeriod",
        "meanPeriod",
        "windSpeed",
        "windDirection",
        "surfaceTemperature",
        "batteryVoltage",
    ]

    for name in copy_vars:
        if name in spotter:
            binned[name] = spotter[name]

    if "trajectory" in spotter.coords:
        binned = binned.assign_coords(trajectory=spotter["trajectory"])

    if "rowsize" in spotter:
        binned["rowsize"] = spotter["rowsize"]

    return binned


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude/latitude in degrees to 3D unit-sphere coordinates.
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack([x, y, z])


def nearest_era5_time_indices(
    spotter_time: np.ndarray,
    era5_time_index: pd.DatetimeIndex,
    *,
    tolerance: str = "30min",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest ERA5 time index for each Spotter observation.

    Returns
    -------
    time_idx : np.ndarray
        ERA5 integer time indices for valid matches.
    valid : np.ndarray
        Boolean mask of valid Spotter observations.
    """
    spotter_time_index = pd.DatetimeIndex(spotter_time)

    time_idx = era5_time_index.get_indexer(
        spotter_time_index,
        method="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    valid = time_idx >= 0
    return time_idx[valid], valid


def nearest_era5_space_indices(
    spotter_lat: np.ndarray,
    spotter_lon: np.ndarray,
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest ERA5 spatial grid point for each Spotter observation.

    Uses a 3D unit-sphere KD-tree.

    Returns
    -------
    value_idx : np.ndarray
        ERA5 values-index match for each Spotter observation.
    distance_m : np.ndarray
        Great-circle distance in metres.
    """
    era5_xyz = lonlat_to_xyz(era5_lon, era5_lat)
    spotter_xyz = lonlat_to_xyz(spotter_lon, spotter_lat)

    tree = cKDTree(era5_xyz)
    dist_chord, value_idx = tree.query(spotter_xyz, k=1)

    angular_distance = 2.0 * np.arcsin(np.clip(dist_chord / 2.0, 0.0, 1.0))
    distance_m = EARTH_RADIUS_M * angular_distance

    return value_idx, distance_m


def build_matchup_dataset(
    spotter_binned: xr.Dataset,
    era5_spectrum: xr.DataArray,
    *,
    time_tolerance: str = "30min",
    max_distance_m: float = 50_000.0,
) -> xr.Dataset:
    """
    Build collocated Spotter-ERA5 variance-density matchup dataset.

    Parameters
    ----------
    spotter_binned : xr.Dataset
        Spotter dataset with varianceDensity(index, frequency).
    era5_spectrum : xr.DataArray
        ERA5 variance density with dimensions time, frequency, values.
    time_tolerance : str
        Maximum allowed time difference for nearest ERA5 match.
    max_distance_m : float
        Maximum allowed distance to nearest ERA5 grid cell.

    Returns
    -------
    xr.Dataset
        Matchup dataset with variables:
        varianceDensity_spotter(index, frequency)
        varianceDensity_era5(index, frequency)
        spotter_latitude(index)
        spotter_longitude(index)
        era5_latitude(index)
        era5_longitude(index)
        era5_distance_m(index)
        era5_time(index)
    """
    required_spotter_vars = ["time", "latitude", "longitude", "varianceDensity"]

    for name in required_spotter_vars:
        if name not in spotter_binned:
            raise KeyError(f"spotter_binned is missing {name!r}.")

    for dim in ["time", "frequency", "values"]:
        if dim not in era5_spectrum.dims:
            raise ValueError(f"era5_spectrum must have dimension {dim!r}.")

    # Remove observations without valid position.
    spotter_common = spotter_binned.where(
        np.isfinite(spotter_binned["latitude"])
        & np.isfinite(spotter_binned["longitude"]),
        drop=True,
    )

    # Restrict Spotter to ERA5 temporal coverage.
    spotter_common = spotter_common.where(
        (spotter_common["time"] >= era5_spectrum["time"].min())
        & (spotter_common["time"] <= era5_spectrum["time"].max()),
        drop=True,
    )

    # Match time.
    time_idx, valid_time = nearest_era5_time_indices(
        spotter_common["time"].values,
        era5_spectrum.indexes["time"],
        tolerance=time_tolerance,
    )

    spotter_common = spotter_common.isel(index=valid_time)

    matched_era5_time = era5_spectrum["time"].values[time_idx]

    spotter_common = spotter_common.assign_coords(
        era5_time=("index", matched_era5_time),
    )

    # Match space.
    era5_lat = era5_spectrum["latitude"].compute().values
    era5_lon = era5_spectrum["longitude"].compute().values

    value_idx, distance_m = nearest_era5_space_indices(
        spotter_common["latitude"].values,
        spotter_common["longitude"].values,
        era5_lat,
        era5_lon,
    )

    spotter_common = spotter_common.assign(
        era5_value_index=("index", value_idx),
        era5_distance_m=("index", distance_m),
        era5_latitude=("index", era5_lat[value_idx]),
        era5_longitude=("index", era5_lon[value_idx]),
    )

    # Apply distance threshold.
    valid_space = spotter_common["era5_distance_m"] <= max_distance_m

    spotter_common = spotter_common.where(valid_space, drop=True)

    time_idx = time_idx[valid_space.values]
    value_idx = value_idx[valid_space.values]

    # Vectorized ERA5 extraction.
    time_indexer = xr.DataArray(
        time_idx,
        dims="index",
        coords={"index": spotter_common["index"]},
    )

    value_indexer = xr.DataArray(
        value_idx,
        dims="index",
        coords={"index": spotter_common["index"]},
    )

    era5_matched = era5_spectrum.isel(
        time=time_indexer,
        values=value_indexer,
    )

    era5_matched = era5_matched.transpose("index", "frequency")
    era5_matched = era5_matched.rename("varianceDensity_era5")

    era5_matched = era5_matched.drop_vars(
        ["latitude", "longitude", "valid_time"],
        errors="ignore",
    )

    spotter_matched = spotter_common["varianceDensity"].transpose(
        "index",
        "frequency",
    )
    spotter_matched = spotter_matched.rename("varianceDensity_spotter")
    spotter_matched = spotter_matched.drop_vars(
        ["latitude", "longitude"],
        errors="ignore",
    )

    matchup = xr.Dataset(
        data_vars={
            "varianceDensity_spotter": spotter_matched,
            "varianceDensity_era5": era5_matched,
            "spotter_latitude": ("index", spotter_common["latitude"].values),
            "spotter_longitude": ("index", spotter_common["longitude"].values),
            "era5_latitude": ("index", spotter_common["era5_latitude"].values),
            "era5_longitude": ("index", spotter_common["era5_longitude"].values),
            "era5_distance_m": ("index", spotter_common["era5_distance_m"].values),
            "era5_time": ("index", spotter_common["era5_time"].values),
        },
        coords={
            "time": ("index", spotter_common["time"].values),
            "frequency": spotter_common["frequency"].values,
        },
        attrs={
            "description": "Collocated Spotter and ERA5 variance-density spectra",
            "time_tolerance": time_tolerance,
            "max_distance_m": max_distance_m,
        },
    )

    return matchup


def spectral_frequency_widths(frequency: xr.DataArray | np.ndarray) -> xr.DataArray:
    """
    Return frequency-bin widths for an ERA5-style logarithmic grid.
    """
    if isinstance(frequency, xr.DataArray):
        freq_values = frequency.values
        coords = {"frequency": frequency}
    else:
        freq_values = np.asarray(frequency)
        coords = {"frequency": freq_values}

    edges = frequency_edges_from_centres(
        freq_values,
        logarithmic=True,
        ratio=1.1,
    )

    return xr.DataArray(
        np.diff(edges),
        dims="frequency",
        coords=coords,
        name="df",
        attrs={"units": "Hz"},
    )


def compute_variance_density_metrics(
    matchup: xr.Dataset,
    *,
    eps: float = 1e-10,
) -> xr.Dataset:
    """
    Compute Spotter-ERA5 variance-density comparison metrics.

    Metrics are calculated frequency-by-frequency over the index dimension.
    """
    obs = matchup["varianceDensity_spotter"]
    mod = matchup["varianceDensity_era5"]

    valid = (
        np.isfinite(obs)
        & np.isfinite(mod)
        & (obs >= 0)
        & (mod >= 0)
    )

    obs_v = obs.where(valid)
    mod_v = mod.where(valid)

    diff = mod_v - obs_v

    mean_obs = obs_v.mean("index")
    mean_mod = mod_v.mean("index")

    bias = diff.mean("index")
    mae = np.abs(diff).mean("index")
    rmse = np.sqrt((diff**2).mean("index"))
    correlation = xr.corr(mod_v, obs_v, dim="index")

    relative_bias = bias / mean_obs
    energy_ratio = mean_mod / mean_obs
    scatter_index = rmse / mean_obs

    obs_pos = obs_v.where(obs_v > eps)
    mod_pos = mod_v.where(mod_v > eps)

    log_obs = np.log10(obs_pos)
    log_mod = np.log10(mod_pos)
    log_diff = log_mod - log_obs

    log10_bias = log_diff.mean("index")
    log10_rmse = np.sqrt((log_diff**2).mean("index"))
    log10_correlation = xr.corr(log_mod, log_obs, dim="index")

    df = spectral_frequency_widths(matchup["frequency"])

    m0_obs = (obs_v * df).sum("frequency")
    m0_mod = (mod_v * df).sum("frequency")

    hs_obs = 4.0 * np.sqrt(m0_obs)
    hs_mod = 4.0 * np.sqrt(m0_mod)

    valid_hs = np.isfinite(hs_obs) & np.isfinite(hs_mod)

    hs_obs = hs_obs.where(valid_hs)
    hs_mod = hs_mod.where(valid_hs)

    hs_bias = (hs_mod - hs_obs).mean("index")
    hs_mae = np.abs(hs_mod - hs_obs).mean("index")
    hs_rmse = np.sqrt(((hs_mod - hs_obs) ** 2).mean("index"))
    hs_correlation = xr.corr(hs_mod, hs_obs, dim="index")

    metrics = xr.Dataset(
        data_vars={
            "mean_spectrum_spotter": mean_obs,
            "mean_spectrum_era5": mean_mod,
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "relative_bias": relative_bias,
            "energy_ratio": energy_ratio,
            "scatter_index": scatter_index,
            "log10_bias": log10_bias,
            "log10_rmse": log10_rmse,
            "log10_correlation": log10_correlation,
            "hs_bias": hs_bias,
            "hs_mae": hs_mae,
            "hs_rmse": hs_rmse,
            "hs_correlation": hs_correlation,
        },
        attrs={
            "description": "Spotter-ERA5 variance-density comparison metrics",
            "positive_floor_for_log10": eps,
        },
    )

    return metrics


def plot_mean_spectra(metrics: xr.Dataset):
    """
    Plot mean Spotter and ERA5 variance-density spectra.

    Returns
    -------
    matplotlib.axes.Axes
        Plot axes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        metrics["frequency"],
        metrics["mean_spectrum_spotter"],
        label="Spotter",
    )

    ax.loglog(
        metrics["frequency"],
        metrics["mean_spectrum_era5"],
        label="ERA5",
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Variance density [m² Hz⁻¹]")
    ax.set_title("Mean variance density spectrum")
    ax.grid(True, which="both")
    ax.legend()

    return ax


def _update_config_from_kwargs(
    config: MatchupConfig,
    **kwargs: Any,
) -> MatchupConfig:
    """
    Return a copy of config with non-None keyword values replaced.
    """
    updates = {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }

    if not updates:
        return config

    return replace(config, **updates)


def run_workflow(
    config: MatchupConfig | None = None,
    *,
    start_time: str | np.datetime64 | pd.Timestamp | None = None,
    end_time: str | np.datetime64 | pd.Timestamp | None = None,
    time_tolerance: str | None = None,
    max_distance_m: float | None = None,
    spotter_chunks: dict[str, int] | None = None,
    era5_chunks: dict[str, int] | None = None,
    era5_spectrum_chunks: dict[str, int] | None = None,
    output_matchup_zarr: str | None = None,
    output_metrics_netcdf: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Run complete Spotter-ERA5 spectral comparison workflow.

    You can either pass a MatchupConfig object or pass keyword overrides
    directly to this function.

    Examples
    --------
    config = MatchupConfig(start_time="2022-01-01", end_time="2022-01-07")
    matchup, metrics = run_workflow(config)

    matchup, metrics = run_workflow(
        start_time="2022-01-01",
        end_time="2022-01-07",
    )
    """
    if config is None:
        config = MatchupConfig()

    config = _update_config_from_kwargs(
        config,
        start_time=start_time,
        end_time=end_time,
        time_tolerance=time_tolerance,
        max_distance_m=max_distance_m,
        spotter_chunks=spotter_chunks,
        era5_chunks=era5_chunks,
        era5_spectrum_chunks=era5_spectrum_chunks,
        output_matchup_zarr=output_matchup_zarr,
        output_metrics_netcdf=output_metrics_netcdf,
    )

    spotter, era5 = open_datasets(config)

    # Subset by time as early as possible.
    spotter = subset_time_range(
        spotter,
        config.start_time,
        config.end_time,
        time_name="time",
    )

    era5 = subset_time_range(
        era5,
        config.start_time,
        config.end_time,
        time_name="time",
    )

    # Apply chunking.
    spotter = ensure_chunked(spotter, config.spotter_chunks)
    era5 = ensure_chunked(era5, config.era5_chunks)

    # Build spectra.
    spotter_binned = build_spotter_binned_dataset(spotter)

    era5_spectrum = integrate_era5_to_1d_spectrum(era5)
    era5_spectrum = ensure_chunked(
        era5_spectrum,
        config.era5_spectrum_chunks,
    )

    # Build matchup.
    matchup = build_matchup_dataset(
        spotter_binned,
        era5_spectrum,
        time_tolerance=config.time_tolerance,
        max_distance_m=config.max_distance_m,
    )

    # Compute metrics lazily.
    metrics = compute_variance_density_metrics(matchup)

    if config.output_matchup_zarr is not None:
        matchup.to_zarr(config.output_matchup_zarr, mode="w")

    if config.output_metrics_netcdf is not None:
        metrics.to_netcdf(config.output_metrics_netcdf)

    return matchup, metrics


if __name__ == "__main__":
    config = MatchupConfig(
        start_time="2022-01-01",
        end_time="2022-01-07",
        time_tolerance="30min",
        max_distance_m=50_000,
        spotter_chunks={"index": 20_000, "frequency": -1},
        era5_spectrum_chunks={"time": 24, "frequency": -1},
    )

    matchup_ds, metrics_ds = run_workflow(config)

    print(matchup_ds)
    print(metrics_ds)

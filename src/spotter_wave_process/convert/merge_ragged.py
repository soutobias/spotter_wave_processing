from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

def normalise_spotter_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}

    if "variance_density" in ds:
        rename["variance_density"] = "varianceDensity"

    if "u10" in ds:
        rename["u10"] = "windSpeed"

    return ds.rename(rename)

def trajectory_id_per_observation(ds: xr.Dataset) -> np.ndarray:
    rowsize = ds["rowsize"].values.astype(np.int64)
    trajectory = ds["trajectory"].values.astype(object)

    if rowsize.sum() != ds.sizes["index"]:
        raise ValueError(
            f"rowsize sum {rowsize.sum()} does not match index size {ds.sizes['index']}"
        )

    return np.repeat(trajectory, rowsize)


def ragged_to_observation_dataset(ds: xr.Dataset) -> xr.Dataset:
    trajectory_id = trajectory_id_per_observation(ds)

    # Remove ragged representation temporarily.
    out = ds.drop_vars(["rowsize"], errors="ignore")

    # Drop the old trajectory coordinate/dimension.
    if "trajectory" in out.coords:
        out = out.drop_vars("trajectory")

    out = out.assign_coords(
        trajectoryId=("index", trajectory_id)
    )

    return out

def filter_bbox(
    ds: xr.Dataset,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )

    return ds.isel(index=valid)

def get_time_limits_by_trajectory(existing_obs: xr.Dataset) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "trajectoryId": existing_obs["trajectoryId"].values.astype(str),
            "time": pd.to_datetime(existing_obs["time"].values),
        }
    )

    limits = (
        meta.groupby("trajectoryId")["time"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "old_start", "max": "old_end"})
    )

    return limits

def filter_out_existing_time_ranges(
    candidate_obs: xr.Dataset,
    existing_limits: pd.DataFrame,
) -> xr.Dataset:
    candidate_meta = pd.DataFrame(
        {
            "trajectoryId": candidate_obs["trajectoryId"].values.astype(str),
            "time": pd.to_datetime(candidate_obs["time"].values),
        }
    )

    merged = candidate_meta.merge(
        existing_limits,
        on="trajectoryId",
        how="left",
    )

    is_new_trajectory = merged["old_start"].isna()

    is_outside_existing_range = (
        (merged["time"] < merged["old_start"])
        | (merged["time"] > merged["old_end"])
    )

    keep = (is_new_trajectory | is_outside_existing_range).to_numpy()

    return candidate_obs.isel(index=keep)

def observation_to_ragged_dataset(ds: xr.Dataset) -> xr.Dataset:
    meta = pd.DataFrame(
        {
            "trajectoryId": ds["trajectoryId"].values.astype(str),
            "time": pd.to_datetime(ds["time"].values),
            "original_index": np.arange(ds.sizes["index"]),
        }
    )

    meta = meta.sort_values(["trajectoryId", "time"]).reset_index(drop=True)
    order = meta["original_index"].to_numpy()

    ds = ds.isel(index=order)

    trajectory = meta["trajectoryId"].drop_duplicates().to_numpy(dtype=object)

    rowsize = (
        meta.groupby("trajectoryId", sort=False)
        .size()
        .to_numpy(dtype=np.int64)
    )

    ds = ds.drop_vars("trajectoryId")

    ds = ds.assign_coords(
        trajectory=("trajectory", trajectory)
    )

    ds["rowsize"] = ("trajectory", rowsize)

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

    return ds

def make_zarr_encoding(ds, *, index_chunksize=500_000):
    encoding = {}

    for name in ds.data_vars:
        dims = ds[name].dims

        if dims == ("index",):
            encoding[name] = {
                "chunks": (index_chunksize,),
            }

        elif dims == ("index", "frequency"):
            encoding[name] = {
                "chunks": (index_chunksize, ds.sizes["frequency"]),
            }

        elif dims == ("trajectory",):
            encoding[name] = {
                "chunks": (ds.sizes["trajectory"],),
            }

    if "time" in ds.coords:
        encoding["time"] = {
            "chunks": (index_chunksize,),
            "dtype": "int64",
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "standard",
        }

    if "frequency" in ds.coords:
        encoding["frequency"] = {
            "chunks": (ds.sizes["frequency"],),
        }

    if "trajectory" in ds.coords:
        encoding["trajectory"] = {
            "chunks": (ds.sizes["trajectory"],),
        }

    return encoding

def merge_spotter_zarrs(
    existing_zarr: str | Path,
    bulk_zarr: str | Path,
    spectral_zarr: str | Path,
    output_zarr: str | Path,
    *,
    lat_min: float = -74.0,
    lat_max: float = -39.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    index_chunksize: int = 50_000,
) -> xr.Dataset:
    existing = xr.open_zarr(existing_zarr, consolidated=True)
    bulk = xr.open_zarr(bulk_zarr, consolidated=True)
    spectral = xr.open_zarr(spectral_zarr, consolidated=True)

    existing = normalise_spotter_names(existing)
    bulk = normalise_spotter_names(bulk)
    spectral = normalise_spotter_names(spectral)

    existing_obs = ragged_to_observation_dataset(existing)
    bulk_obs = ragged_to_observation_dataset(bulk)
    spectral_obs = ragged_to_observation_dataset(spectral)

    bulk_obs = filter_bbox(
        bulk_obs,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    spectral_obs = filter_bbox(
        spectral_obs,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    existing_limits = get_time_limits_by_trajectory(existing_obs)

    bulk_new = filter_out_existing_time_ranges(
        bulk_obs,
        existing_limits,
    )

    spectral_new = filter_out_existing_time_ranges(
        spectral_obs,
        existing_limits,
    )
    merged_obs = xr.concat(
        [existing_obs, bulk_new, spectral_new],
        dim="index",
        join="outer",
        compat="override",
        coords="minimal",
        data_vars="all",
        fill_value=np.nan,
    )

    merged = observation_to_ragged_dataset(merged_obs)

    merged.attrs.update(existing.attrs)
    merged.attrs["title"] = "Merged Spotter wave buoy ragged dataset"
    merged.attrs["merged_from"] = (
        "existing dataset, Sofar bulk wave parameters, "
        "Sofar spectral wave parameters"
    )

    index_chunksize = 150_000

    encoding = make_zarr_encoding(
        merged,
        index_chunksize=index_chunksize,
    )


    chunk_map = {
        "index": index_chunksize,
    }

    if "frequency" in merged.dims:
        chunk_map["frequency"] = merged.sizes["frequency"]

    if "trajectory" in merged.dims:
        chunk_map["trajectory"] = merged.sizes["trajectory"]

    merged = merged.chunk(chunk_map)

    merged.to_zarr(
        output_zarr,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    return merged

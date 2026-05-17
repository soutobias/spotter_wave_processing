#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cfgrib
import xarray as xr


LOGGER = logging.getLogger(__name__)


def get_storage_options(credentials_path: str | None):
    if credentials_path is None or credentials_path.lower() in {"none", "null", "-"}:
        return None

    with open(credentials_path) as f:
        credentials = json.load(f)

    return {
        "key": credentials["token"],
        "secret": credentials["secret"],
        "client_kwargs": {
            "endpoint_url": credentials["endpoint_url"],
        },
        "config_kwargs": {
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required",
        },
    }



def normalise_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare one cfgrib dataset before writing to Zarr.
    """
    # Make sure time is monotonic.
    if "time" in ds.coords:
        ds = ds.sortby("time")

    # valid_time is often identical to time when step == 0.
    # Keeping it is fine, but it can sometimes complicate appends.
    # Uncomment this if append errors appear because of valid_time.
    #
    # if "valid_time" in ds.coords:
    #     ds = ds.drop_vars("valid_time")

    # Remove duplicated timestamps within a single file, if any.
    if "time" in ds.indexes:
        _, unique_index = xr.DataArray(ds.time).to_index().duplicated(), None
        ds = ds.sel(time=~ds.get_index("time").duplicated())

    return ds


def chunk_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Chunk all variables with:
      - time: 1
      - latitude: full dimension
      - longitude: full dimension
    """
    chunks: dict[str, int] = {}

    for dim, size in ds.sizes.items():
        if dim == "time":
            chunks[dim] = 1
        elif dim in {"latitude", "longitude"}:
            chunks[dim] = size
        else:
            chunks[dim] = size

    return ds.chunk(chunks)


def zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """
    Force Zarr chunk layout to match the Dask chunking.
    """
    encoding: dict[str, dict[str, Any]] = {}

    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim in {"latitude", "longitude"}:
                chunks.append(ds.sizes[dim])
            else:
                chunks.append(ds.sizes[dim])

        encoding[name] = {
            "chunks": tuple(chunks),
        }

    return encoding


def infer_group_name(ds: xr.Dataset, fallback_index: int) -> str:
    """
    Give stable, readable names to the two cfgrib groups.
    """
    variables = set(ds.data_vars)

    if {"swh", "mwd", "pp1d", "mwp"}.issubset(variables):
        return "group_0_wave"

    if {"sst", "u10", "v10"}.issubset(variables):
        return "group_1_atmos"

    return f"group_{fallback_index}"


def write_or_append_zarr(
    ds: xr.Dataset,
    output_path: str,
    storage_options: dict[str, Any] | None,
    first_write: bool,
) -> None:
    """
    Write first year with mode='w', then append later years along time.
    """
    ds = chunk_dataset(ds)
    encoding = zarr_encoding(ds)

    kwargs = dict(
        store=output_path,
        consolidated=True,
        storage_options=storage_options,
        compute=True,
    )

    if first_write:
        LOGGER.info("Writing new Zarr store: %s", output_path)
        ds.to_zarr(
            mode="w",
            encoding=encoding,
            **kwargs,
        )
    else:
        LOGGER.info("Appending to Zarr store: %s", output_path)
        ds.to_zarr(
            mode="a",
            append_dim="time",
            **kwargs,
        )


def open_grib_groups(path: Path) -> list[xr.Dataset]:
    """
    Open all compatible cfgrib groups from one GRIB file.

    indexpath='' avoids stale .idx files on shared filesystems.
    """
    LOGGER.info("Opening %s", path)

    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={
            "indexpath": "",
        },
    )

    return [normalise_dataset(ds) for ds in datasets]


def convert_years_to_zarr(
    input_dir: Path,
    output_base: str,
    years: list[int],
    credentials_path: str | None,
) -> None:
    storage_options = get_storage_options(credentials_path)

    written_groups: dict[str, bool] = {}

    for year in years:
        grib_path = input_dir / f"parametric_data_{year}.grib"

        if not grib_path.exists():
            raise FileNotFoundError(f"Missing GRIB file: {grib_path}")

        datasets = open_grib_groups(grib_path)

        if len(datasets) != 2:
            raise ValueError(
                f"Expected 2 datasets from {grib_path}, got {len(datasets)}"
            )

        for i, ds in enumerate(datasets):
            group_name = infer_group_name(ds, i)
            output_path = f"{output_base.rstrip('/')}/{group_name}.zarr"

            LOGGER.info(
                "Year=%s group=%s vars=%s dims=%s",
                year,
                group_name,
                list(ds.data_vars),
                dict(ds.sizes),
            )

            first_write = not written_groups.get(group_name, False)

            write_or_append_zarr(
                ds=ds,
                output_path=output_path,
                storage_options=storage_options,
                first_write=first_write,
            )

            written_groups[group_name] = True

            ds.close()

    LOGGER.info("Finished writing all years.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert yearly parametric GRIB files to chunked Zarr stores."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing parametric_data_YYYY.grib files.",
    )

    parser.add_argument(
        "--output-base",
        type=str,
        required=True,
        help=(
            "Object-store base path, for example "
            "s3://my-bucket/waves/parametric_data"
        ),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--credentials-path",
        type=str,
        default=None,
        help="Optional JSON file with fsspec storage_options.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()

    years = list(range(args.start_year, args.end_year + 1))

    convert_years_to_zarr(
        input_dir=args.input_dir,
        output_base=args.output_base,
        years=years,
        credentials_path=args.credentials_path,
    )


if __name__ == "__main__":
    main()

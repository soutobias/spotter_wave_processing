""" Download ocean files from Met Office FTP server.
"""

import gc

import cartopy.crs as ccrs
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from spotter_wave_process.qc import SpotterQARTODQC
from pathlib import Path
import plotly.graph_objects as go


SPOTTER_URL = "https://paidiver-o.s3-ext.jc.rl.ac.uk/waves/spotter_brazil.zarr"
ERA5_URL = "https://paidiver-o.s3-ext.jc.rl.ac.uk/waves/era3/waves.zarr"
GFS_URL = "https://erddap.aoml.noaa.gov/hdb/erddap/griddap/WaveWatch_"

class SpotterProcess:

    def __init__(
        self,
        output_path: str = "data",
        n_workers: int = 16,
        lons: list = [-180, 180],
        lats: list = [-67.5, -50.5],
        start_date: str = "2021-12-01",
        end_date: str = "2021-12-10",
        model_sources: list[str] | None = None,
        era5_path: str | None = None,
    ):
        self.output_path = f"{os.getcwd()}/{output_path}"
        self.n_workers = n_workers
        self.lons = lons
        self.lats = lats
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.model_sources = model_sources or ["era5"]
        self.era5_path = era5_path or ERA5_URL

        self.var_list = ["latitude", "longitude", "time", "significantWaveHeight", "peakPeriod"]
        self.model_variable_map = {
            "gfs": {
                "significantWaveHeight": "Thgt",
                "peakPeriod": "Tper",
            },
            "era5": {
                "significantWaveHeight": "swh",
                "peakPeriod": "pp1d",
            },
        }
        self.model_display_names = {
            "gfs": "NOAA GFS",
            "era5": "ERA5",
        }
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):
        self.model_dss = self.download_model_data()

        spotter_df_dict = self.download_spotter_data()
        self.spotter_df = self.interpolate_spotter(spotter_df_dict)

        self.spotter_df = self.merge_with_models()
        self.spotter_df = self.qc_spotter_data(self.spotter_df)

        self.bias_by_hour = self.calculate_bias_by_hour()

    def qc_spotter_data(self, spotter_df):
        qc = SpotterQARTODQC(spotter_df)
        return qc.run_all(model_comparison_can_fail=True, remove_failed_after=True)

    def download_model_data(self):
        model_dss = {}

        for source in self.model_sources:
            if source == "gfs":
                model_dss[source] = self.download_gfs_data()
            elif source == "era5":
                model_dss[source] = self.download_era5_data()
            else:
                raise ValueError(f"Unknown model source: {source}")

        return model_dss

    def download_era5_data(self):
        """
        Load ERA5 wave reanalysis data from a local GRIB file.

        Expected ERA5 variables:
            swh  = significant wave height
            pp1d = peak wave period from 1D spectra
            mwp  = mean wave period
            mwd  = mean wave direction
        """
        if self.era5_path is None:
            raise ValueError("era5_path must be provided when using model_sources=['era5'].")

        era5_ds = xr.open_zarr(
            self.era5_path,
        )

        wanted_vars = ["swh", "pp1d"]
        missing_vars = [var for var in wanted_vars if var not in era5_ds]

        if missing_vars:
            raise ValueError(
                f"ERA5 dataset is missing required variables: {missing_vars}. "
                f"Available variables are: {list(era5_ds.data_vars)}"
            )

        era5_ds = era5_ds[wanted_vars]

        era5_ds = era5_ds.sel(
            time=slice(
                self.start_date.strftime("%Y-%m-%d"),
                self.end_date.strftime("%Y-%m-%d"),
            )
        )

        # ERA5 latitude is descending in your file: -39, -39.5, ..., -74.
        # xarray slice must follow coordinate order.
        lat_values = era5_ds.latitude.values

        if lat_values[0] > lat_values[-1]:
            lat_slice = slice(self.lats[1], self.lats[0])
        else:
            lat_slice = slice(self.lats[0], self.lats[1])

        era5_ds = era5_ds.sel(
            latitude=lat_slice,
            longitude=slice(self.lons[0], self.lons[1]),
        )

        return era5_ds


    def calculate_bias_by_hour(self):
        results = {}

        timestamps = self.spotter_df["timestamp"].dt.tz_convert("UTC")
        start_ts = pd.Timestamp(self.start_date, tz="UTC")

        self.spotter_df = self.spotter_df.assign(
            analysis_hour=((timestamps - start_ts).dt.total_seconds() // 3600).astype(int)
        )

        for model_name in self.model_sources:
            results[model_name] = {}

            variable_map = self.model_variable_map[model_name]

            for spotter_var, model_var in variable_map.items():
                value_col = f"model_value_{model_name}_{model_var}"
                error_col = f"error_{model_name}_{model_var}"

                agg_df = (
                    self.spotter_df
                    .groupby("analysis_hour")
                    .agg(
                        bias=(error_col, "mean"),
                        model_value=(value_col, "mean"),
                        data_value=(spotter_var, "mean"),
                    )
                    .reset_index()
                    .sort_values("analysis_hour")
                )

                results[model_name][model_var] = agg_df

        return results


    def get_model_collocation(self, curr_time, model_da, spotter_df):
        try:
            curr_time = pd.Timestamp(curr_time).tz_convert("UTC")

            curr_model_da = model_da.sel(
                time=curr_time.to_datetime64(),
                method="nearest",
            )

            curr_spotter_df = spotter_df[spotter_df["timestamp"] == curr_time]

            if curr_spotter_df.empty:
                return curr_time, None

            collocated_vals = curr_model_da.interp(
                latitude=xr.DataArray(
                    curr_spotter_df.latitude.values,
                    dims="points",
                ),
                longitude=xr.DataArray(
                    curr_spotter_df.longitude.values,
                    dims="points",
                ),
                method="linear",
            ).values

            return curr_time, collocated_vals

        except Exception as e:
            print(f"Error at time {curr_time}: {e}")
            return curr_time, None

    def merge_with_models(self):
        """
        Collocate Spotter observations with all configured model datasets.
        """
        for model_name, model_ds in self.model_dss.items():
            variable_map = self.model_variable_map[model_name]

            for spotter_var, model_var in variable_map.items():
                model_da = model_ds[model_var]

                value_col = f"model_value_{model_name}_{model_var}"
                error_col = f"error_{model_name}_{model_var}"

                self.spotter_df[value_col] = np.nan

                unique_times = pd.to_datetime(
                    self.spotter_df["timestamp"].unique(),
                    utc=True,
                )

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    future_to_time = {
                        executor.submit(
                            self.get_model_collocation,
                            curr_time,
                            model_da,
                            self.spotter_df,
                        ): curr_time
                        for curr_time in unique_times
                    }

                    desc = f"Collocating {model_name.upper()} {model_var}"

                    for future in tqdm(
                        as_completed(future_to_time),
                        total=len(future_to_time),
                        desc=desc,
                    ):
                        curr_time, collocated_vals = future.result()

                        if collocated_vals is not None:
                            mask = self.spotter_df["timestamp"] == curr_time
                            self.spotter_df.loc[mask, value_col] = collocated_vals

                self.spotter_df[error_col] = (
                    self.spotter_df[value_col] - self.spotter_df[spotter_var]
                )

        return self.spotter_df

    def process_interpolation(self, spotter_id, curr_data):
        try:
            new_df = pd.DataFrame()

            earliest_interp_time = (
                curr_data.loc[curr_data.index.values[0], ["time"]].item().ceil("h")
            )
            latest_interp_time = (
                curr_data.loc[curr_data.index.values[-1], ["time"]].item().floor("h")
            )

            new_df["time"] = pd.date_range(
                start=earliest_interp_time, end=latest_interp_time, freq="h"
            )

            interp_times = [
                t.to_datetime64().astype("int64") // 1e9 for t in new_df["time"]
            ]

            for var in self.var_list:
                if var == "longitude":
                    f_interp = interp1d(
                        [
                            t.to_datetime64().astype("int64") // 1e9
                            for t in curr_data["time"]
                        ],
                        curr_data[var],
                        kind="nearest",
                    )
                else:
                    f_interp = interp1d(
                        [
                            t.to_datetime64().astype("int64") // 1e9
                            for t in curr_data["time"]
                        ],
                        curr_data[var],
                        kind="linear",
                    )

                new_df[var] = f_interp(interp_times)

            new_df["spotter_id"] = spotter_id
            return new_df

        except Exception as e:
            print(f"Error interpolating Spotter {spotter_id}: {e}")
            print(curr_data[["time", "latitude", "longitude"]].head())
            print(curr_data[["time", "latitude", "longitude"]].tail())
            return None


    def interpolate_spotter(self, spotter_df_dict):
        new_df_list = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_spotter = {
                executor.submit(self.process_interpolation, spotter_id, curr_data): spotter_id
                for spotter_id, curr_data in spotter_df_dict.items()
            }

            for future in tqdm(as_completed(future_to_spotter), total=len(future_to_spotter), desc="Interpolating Spotters"):
                result = future.result()
                if result is not None:
                    new_df_list.append(result)


        spotter_df = pd.concat(new_df_list).sort_values(by="time", ignore_index=True)
        spotter_df["timestamp"] = pd.to_datetime(spotter_df["time"], utc=True)

        print("AFTER INTERPOLATION")
        print("lon min/max:", spotter_df.longitude.min(), spotter_df.longitude.max())
        print("lat min/max:", spotter_df.latitude.min(), spotter_df.latitude.max())
        print("n rows:", len(spotter_df))
        print("n spotters:", spotter_df.spotter_id.nunique())



        return spotter_df

    def download_spotter_data(self):
        """ Download Spotter data from AWS S3 bucket.
        """
        self.spotter_ds = xr.open_dataset(SPOTTER_URL, engine="zarr")

        return self.process_all_spotters_threaded()

    def download_gfs_data(self):
        """ Download GFS data from AWS S3 bucket.
        """
        years = ["2021", "2022"]
        dss = []
        for year in years:
            url = f"{GFS_URL}{year}"
            noaa_ds = xr.open_dataset(url)
            noaa_ds = noaa_ds[['Tper', 'Thgt']].sel(
                time=slice(self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")),
                latitude=slice(self.lats[0], self.lats[1]),
                longitude=slice(self.lons[0], self.lons[1])
            )
            dss.append(noaa_ds)
        return xr.concat(dss, dim="time")

    def process_spotter(self, sp_idx, spotter_id, traj_idx, start_npy, end_npy):
        try:
            spotter_id_str = spotter_id.item() if hasattr(spotter_id, 'item') else spotter_id

            sli = slice(traj_idx[sp_idx], traj_idx[sp_idx+1])

            curr_df = pd.DataFrame({
                "latitude": self.spotter_ds.latitude[sli],
                "longitude": self.spotter_ds.longitude[sli],
                "time": self.spotter_ds.time[sli],
                "significantWaveHeight": self.spotter_ds.significantWaveHeight[sli],
                "peakPeriod": self.spotter_ds.peakPeriod[sli],
            })

            curr_df.sort_values(by=["time"], inplace=True)
            curr_df = curr_df[
                (curr_df["time"].between(start_npy, end_npy)) &
                (curr_df["longitude"].between(self.lons[0], self.lons[1])) &
                (curr_df["latitude"].between(self.lats[0], self.lats[1]))
            ]
            if len(curr_df) <= 1 and len(curr_df) > 0:
                print(
                    "Dropped sparse spotter:",
                    spotter_id_str,
                    "n=", len(curr_df),
                    "lon=", curr_df.longitude.min(), curr_df.longitude.max(),
                    "lat=", curr_df.latitude.min(), curr_df.latitude.max(),
                )

            if len(curr_df) > 1:
                return (spotter_id_str, curr_df)
            else:
                return (spotter_id_str, None)


        except Exception as e:
            print(f"Error processing Spotter {spotter_id}: {str(e)}")
            return (spotter_id, None)

    def process_all_spotters_threaded(self):
        spotter_df_dict = {}

        traj_idx = np.insert(np.cumsum(self.spotter_ds.rowsize.values), 0, 0)
        start_npy = pd.Timestamp(self.start_date-timedelta(minutes=30)).to_numpy()
        end_npy = pd.Timestamp(self.end_date+timedelta(minutes=30)).to_numpy()
        tasks = [
            (sp_idx, spotter_id, traj_idx, start_npy, end_npy)
            for sp_idx, spotter_id in enumerate(self.spotter_ds.trajectory)
        ]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_spotter = {
                executor.submit(self.process_spotter, *task_args): task_args[1]
                for task_args in tasks
            }

            for future in tqdm(as_completed(future_to_spotter), total=len(tasks), desc="Processing Spotters"):
                spotter_id, result_df = future.result()

                if result_df is not None:
                    spotter_df_dict[spotter_id] = result_df


        raw_filtered = pd.concat(
            [df.assign(spotter_id=spotter_id) for spotter_id, df in spotter_df_dict.items()],
            ignore_index=True,
        )

        print("RAW FILTERED AFTER process_spotter")
        print("lon min/max:", raw_filtered.longitude.min(), raw_filtered.longitude.max())
        print("lat min/max:", raw_filtered.latitude.min(), raw_filtered.latitude.max())
        print("n rows:", len(raw_filtered))
        print("n spotters:", raw_filtered.spotter_id.nunique())

        return spotter_df_dict

    def plot_bias(self):
        for model_name, model_results in self.bias_by_hour.items():
            display_name = self.model_display_names.get(model_name, model_name.upper())

            for model_var, bias_by_hour in model_results.items():
                fig, axs = plt.subplots(
                    nrows=1,
                    figsize=(12, 5),
                )

                fig.suptitle(
                    f"Skill of {display_name} compared to Spotter buoys: {model_var} bias"
                )

                axs.set_title(
                    f"{self.start_date.strftime('%Y-%m-%d %HZ')} "
                    f"to {self.end_date.strftime('%Y-%m-%d %HZ')}"
                )

                axs.set_xlabel("Analysis hour")
                axs.set_ylabel("Bias")

                axs.plot(
                    bias_by_hour["analysis_hour"],
                    bias_by_hour["bias"],
                    color="red",
                    markerfacecolor="blue",
                    marker="o",
                    label="Bias",
                )

                axs.plot(
                    bias_by_hour["analysis_hour"],
                    bias_by_hour["model_value"],
                    color="black",
                    label=f"{display_name} value",
                )

                axs.plot(
                    bias_by_hour["analysis_hour"],
                    bias_by_hour["data_value"],
                    color="green",
                    label="Spotter value",
                )

                axs.grid(True)
                axs.legend()

                output_file = f"{self.output_path}/timeseries_{model_name}_{model_var}.png"
                fig.savefig(output_file)
                plt.close(fig)

                print(f"Saved bias timeseries to {output_file}")

    def plot_map(self, model_names=None, model_vars=None, overwrite=False):
        unique_times = pd.to_datetime(
            self.spotter_df["timestamp"].unique(),
            utc=True,
        )
        unique_times = np.sort(unique_times)

        if not model_names:
            model_names = self.model_sources
        if not model_vars:
            model_vars = set()
            for model_name in model_names:
                variable_map = self.model_variable_map.get(model_name, {})
                model_vars.update(variable_map.values())
            model_vars = list(model_vars)

        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model_ds in self.model_dss.items():
            if model_name not in model_names:
                continue

            variable_map = self.model_variable_map[model_name]

            for _, model_var in variable_map.items():
                if model_var not in model_vars:
                    continue

                desc = f"Generating maps for {model_name.upper()} {model_var}"

                min_cmap = self.bias_by_hour[model_name][model_var]["bias"].min()
                max_cmap = self.bias_by_hour[model_name][model_var]["bias"].max()
                min_model_cmap = self.bias_by_hour[model_name][model_var]["model_value"].min()
                max_model_cmap = self.bias_by_hour[model_name][model_var]["model_value"].max()

                saved_count = 0
                skipped_count = 0

                for curr_time in tqdm(unique_times, total=len(unique_times), desc=desc):
                    output_file = (
                        output_path
                        / f"2d_errors_{model_name}_{model_var}_"
                        f"{curr_time.strftime('%Y%m%d%H')}.png"
                    )

                    if output_file.exists() and not overwrite:
                        skipped_count += 1
                        continue

                    fig, _ = self.plot_single_frame(
                        curr_time=curr_time,
                        model_name=model_name,
                        model_var=model_var,
                        cmap_limits=(min_cmap, max_cmap),
                        model_cmap_limits=(min_model_cmap, max_model_cmap),
                    )

                    fig.savefig(output_file)
                    plt.close(fig)
                    fig.clf()
                    gc.collect()

                    saved_count += 1

                print(
                    f"Saved {saved_count} new images to {self.output_path} "
                    f"for {model_name.upper()} variable {model_var}. "
                    f"Skipped {skipped_count} existing images."
                )

    def plot_single_frame(self, curr_time, model_name, model_var, cmap_limits, model_cmap_limits):
        curr_time = pd.Timestamp(curr_time).tz_convert("UTC")

        display_name = self.model_display_names.get(model_name, model_name.upper())

        curr_df = self.spotter_df[self.spotter_df["timestamp"] == curr_time]

        model_da = self.model_dss[model_name][model_var].sel(
            time=curr_time.to_datetime64(),
            method="nearest",
        )

        error_col = f"error_{model_name}_{model_var}"
        cbar_min, cbar_max = cmap_limits
        model_min, model_max = model_cmap_limits
        if model_var in ["Thgt", "swh"]:
            model_label = f"{display_name}[m]"
            error_label = "diff[m]"
        else:
            model_label = f"{display_name}[s]"
            error_label = "diff[s]"

        proj = ccrs.PlateCarree()

        fig, axs = plt.subplots(
            nrows=1,
            figsize=(12, 5),
            subplot_kw={"projection": proj},
            # layout="compressed",
        )
        display_time = curr_time.strftime("%Y-%m-%d %HZ")

        # fig.suptitle(f"{display_name} - Spotter: {display_time}, {model_var}")

        axs.set_extent(
            [
                self.lons[0],
                self.lons[1],
                self.lats[0],
                self.lats[1],
            ],
            crs=ccrs.PlateCarree(),
        )

        im = model_da.plot(
            ax=axs,
            cmap="YlOrRd",
            add_colorbar=False,
            transform=proj,
        )
        axs.set_title(f"{display_name} - Spotter: {display_time}, {model_var}")

        sc = axs.scatter(
            x=curr_df["longitude"].values,
            y=curr_df["latitude"].values,
            c=curr_df[error_col].values,
            s=30,
            cmap="RdBu_r",
            edgecolors=mpl_colors.colorConverter.to_rgba("black", alpha=1),
            transform=proj,
        )

        sc.set_clim(cbar_min, cbar_max)
        im.set_clim(model_min, model_max)


        divider = make_axes_locatable(axs)

        cax1 = divider.append_axes(
            "right",
            size="4%",
            pad=0.1,
            axes_class=plt.Axes,
        )

        cax2 = divider.append_axes(
            "right",
            size="4%",
            pad=0.7,
            axes_class=plt.Axes,
        )

        cbar_model = fig.colorbar(
            im,
            cax=cax1,
            orientation="vertical",
            label=model_label,
            ticks=[model_min, model_max],
        )

        cbar_errors = fig.colorbar(
            sc,
            cax=cax2,
            orientation="vertical",
            label=error_label,
            ticks=[cbar_min, cbar_max],
        )

        axs.coastlines(edgecolor="#70706f")

        return fig, axs



    def save_outputs(self, save_nc=False):
        self.spotter_df.to_csv(
            f"{self.output_path}/spotter_data.csv",
            index=False,
        )

        print(f"Saved collocated data to {self.output_path}/spotter_data.csv")

        for model_name, model_results in self.bias_by_hour.items():
            for model_var, bias_by_hour in model_results.items():
                output_file = f"{self.output_path}/bias_by_hour_{model_name}_{model_var}.csv"

                bias_by_hour.to_csv(output_file, index=False)

                print(f"Saved bias by hour to {output_file}")

        if save_nc:
            for model_name, model_ds in self.model_dss.items():
                output_file = f"{self.output_path}/{model_name}_data.nc"

                model_ds.to_netcdf(output_file)

                print(f"Saved {model_name.upper()} data to {output_file}")


    def create_plotly_fig(df):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["analysis_hour"],
                y=df["model_value"],
                mode="lines",
                name="Model value",
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["analysis_hour"],
                y=df["data_value"],
                mode="lines",
                name="Data value",
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["analysis_hour"],
                y=df["bias"],
                mode="lines",
                name="Bias",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Model Value, Data Value, and Bias by Analysis Hour",
            xaxis=dict(title="Analysis hour"),
            yaxis=dict(title="Model / Data value"),
            yaxis2=dict(
                title="Bias",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    def plot_scatter_comparison(x_col, y_col, df, label):
        data = df[[x_col, y_col]].dropna()

        x = data[x_col].to_numpy()
        y = data[y_col].to_numpy()

        # Metrics
        rmse = np.sqrt(np.mean((y - x) ** 2))
        mae = np.mean(np.abs(y - x))

        # Relative MAE, using observations as reference
        rmae = mae / np.mean(np.abs(x))

        # Linear trend line: y = m*x + b
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b

        # Plot
        ax = data.plot.scatter(
            x=x_col,
            y=y_col,
            figsize=(7, 6),
            alpha=0.6,
        )

        ax.plot(x_line, y_line, label=f"Trend: y = {m:.2f}x + {b:.2f}")

        # Optional 1:1 line
        ax.plot(x_line, x_line, linestyle="--", label="1:1 line")

        ax.set_xlabel(f"Spotter - {label}")
        ax.set_ylabel(f"ERA5 - {label}")
        ax.set_title(f"Spotter vs ERA5 - {label}")

        text = (
            f"RMSE = {rmse:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"RMAE = {rmae:.3f}"
        )

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        ax.legend()
        plt.show()

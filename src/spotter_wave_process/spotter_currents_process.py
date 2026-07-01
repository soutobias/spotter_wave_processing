""" Download ocean files from Met Office FTP server.
"""

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.interpolate import interp1d
from spotter_wave_process.qc import SpotterQARTODQC

from spotter_wave_process.spotter_wave_process import SpotterProcess


class SpotterVelocityProcess(SpotterProcess):
    """
    Extension of SpotterProcess that derives Spotter drift velocity from
    quality-controlled trajectory positions.

    The derived velocity is the motion of the drifting Spotter. It should be
    treated as a surface-drift/current proxy rather than a perfect Eulerian
    ocean-current measurement.
    """

    def __init__(
        self,
        output_path: str = "data",
        n_workers: int = 16,
        lons: list[float] | None = None,
        lats: list[float] | None = None,
        start_date: str = "2021-12-01",
        end_date: str = "2021-12-10",
        model_sources: list[str] | None = None,
        era5_path: str | None = None,
        *,
        calculate_velocity: bool = True,
        velocity_max_gap_hours: float = 3.0,
        velocity_max_speed_m_s: float = 3.0,
    ):
        """
        Parameters
        ----------
        model_sources:
            Use ["era5"], ["gfs"], ["era5", "gfs"], or [].

            If [] is used, the workflow downloads/processes/QCs Spotter data
            and calculates velocity without downloading ERA5/GFS.

        calculate_velocity:
            Whether to calculate Spotter drift velocity after QC.

        velocity_max_gap_hours:
            Maximum allowed time gap between trajectory positions used for
            velocity calculation.

        velocity_max_speed_m_s:
            Speed threshold above which derived velocities are flagged as bad.
            Keep this configurable because the best value depends on region,
            instrument behaviour and expected conditions.
        """
        effective_model_sources = ["era5"] if model_sources is None else model_sources

        super().__init__(
            output_path=output_path,
            n_workers=n_workers,
            lons=lons or [-180, 180],
            lats=lats or [-67.5, -50.5],
            start_date=start_date,
            end_date=end_date,
            model_sources=effective_model_sources,
            era5_path=era5_path,
        )

        # Override the parent behaviour so that model_sources=[] is valid.
        self.model_sources = effective_model_sources

        # Interpolated variables. Do not include "time" here because time is
        # created explicitly as the hourly interpolation target.
        self.var_list = [
            "latitude",
            "longitude",
            "significantWaveHeight",
            "peakPeriod",
        ]

        self.calculate_velocity = calculate_velocity
        self.velocity_max_gap_hours = velocity_max_gap_hours
        self.velocity_max_speed_m_s = velocity_max_speed_m_s

        self.geod = Geod(ellps="WGS84")

    def run(self):
        """
        Run the Spotter processing workflow.

        Velocity is calculated after QC because velocity should not be derived
        from positions that may later be removed as bad data.
        """
        if self.model_sources:
            self.model_dss = self.download_model_data()
        else:
            self.model_dss = {}

        spotter_df_dict = self.download_spotter_data()
        self.spotter_df = self.interpolate_spotter(spotter_df_dict)

        if self.model_dss:
            self.spotter_df = self.merge_with_models()

        self.spotter_df = self.qc_spotter_data(self.spotter_df)

        if self.calculate_velocity:
            self.spotter_df = self.calculate_drifter_velocity(
                self.spotter_df,
                max_gap_hours=self.velocity_max_gap_hours,
                max_speed_m_s=self.velocity_max_speed_m_s,
            )

        if self.model_sources:
            self.bias_by_hour = self.calculate_bias_by_hour()
        else:
            self.bias_by_hour = {}

        return self.spotter_df

    def process_interpolation(self, spotter_id, curr_data):
        """
        Override parent interpolation to handle longitude more safely.

        Longitude is unwrapped before interpolation and wrapped back to
        [-180, 180] afterwards. This avoids artificial jumps near the dateline.
        """
        try:
            curr_data = (
                curr_data
                .sort_values("time")
                .dropna(subset=["time", "latitude", "longitude"])
                .drop_duplicates(subset=["time"])
            )

            if len(curr_data) < 2:
                return None

            new_df = pd.DataFrame()

            earliest_interp_time = pd.Timestamp(curr_data["time"].iloc[0]).ceil("h")
            latest_interp_time = pd.Timestamp(curr_data["time"].iloc[-1]).floor("h")

            if earliest_interp_time > latest_interp_time:
                return None

            new_df["time"] = pd.date_range(
                start=earliest_interp_time,
                end=latest_interp_time,
                freq="h",
            )

            source_times = (
                pd.to_datetime(curr_data["time"], utc=True)
                .astype("int64")
                .to_numpy(dtype=float)
                / 1e9
            )

            interp_times = (
                pd.to_datetime(new_df["time"], utc=True)
                .astype("int64")
                .to_numpy(dtype=float)
                / 1e9
            )

            for var in self.var_list:
                if var not in curr_data:
                    new_df[var] = np.nan
                    continue

                values = curr_data[var].to_numpy(dtype=float)

                if var == "longitude":
                    lon_unwrapped = np.rad2deg(
                        np.unwrap(np.deg2rad(values))
                    )

                    f_interp = interp1d(
                        source_times,
                        lon_unwrapped,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                    interpolated_lon = f_interp(interp_times)

                    # Wrap back to [-180, 180].
                    new_df[var] = ((interpolated_lon + 180) % 360) - 180

                else:
                    f_interp = interp1d(
                        source_times,
                        values,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                    new_df[var] = f_interp(interp_times)

            new_df["spotter_id"] = spotter_id

            return new_df

        except Exception as e:
            print(f"Error interpolating Spotter {spotter_id}: {e}")
            print(curr_data[["time", "latitude", "longitude"]].head())
            print(curr_data[["time", "latitude", "longitude"]].tail())
            return None

    def calculate_drifter_velocity(
        self,
        spotter_df: pd.DataFrame,
        *,
        max_gap_hours: float = 3.0,
        max_speed_m_s: float = 3.0,
    ) -> pd.DataFrame:
        """
        Calculate Spotter drift velocity from QCed latitude/longitude/time.

        Output columns
        --------------
        drifter_u_m_s:
            Eastward velocity component.

        drifter_v_m_s:
            Northward velocity component.

        drifter_speed_m_s:
            Horizontal drift speed.

        drifter_direction_to_deg:
            Direction the Spotter moves toward, degrees clockwise from north.

        drifter_velocity_dt_s:
            Time window used for velocity estimate.

        drifter_velocity_distance_m:
            Geodesic distance used for velocity estimate.

        drifter_velocity_qc_flag:
            1 = pass
            4 = failed speed threshold
            9 = not computed
        """
        required_cols = {"spotter_id", "latitude", "longitude", "timestamp"}
        missing_cols = required_cols - set(spotter_df.columns)

        if missing_cols:
            raise ValueError(
                f"Missing required columns for velocity calculation: {missing_cols}"
            )

        df = spotter_df.copy()

        velocity_cols = [
            "drifter_u_m_s",
            "drifter_v_m_s",
            "drifter_speed_m_s",
            "drifter_direction_to_deg",
            "drifter_velocity_dt_s",
            "drifter_velocity_distance_m",
        ]

        for col in velocity_cols:
            df[col] = np.nan

        df["drifter_velocity_qc_flag"] = 9

        max_gap_s = max_gap_hours * 3600.0

        processed_groups = []

        for _, group in df.groupby("spotter_id", sort=False):
            processed_groups.append(
                self._calculate_single_spotter_velocity(
                    group=group,
                    max_gap_s=max_gap_s,
                    max_speed_m_s=max_speed_m_s,
                )
            )

        return pd.concat(processed_groups).sort_index()

    def _calculate_single_spotter_velocity(
        self,
        group: pd.DataFrame,
        max_gap_s: float,
        max_speed_m_s: float,
    ) -> pd.DataFrame:
        group = group.sort_values("timestamp").copy()

        n = len(group)

        if n < 2:
            return group

        lon = group["longitude"].to_numpy(dtype=float)
        lat = group["latitude"].to_numpy(dtype=float)

        t = (
            pd.to_datetime(group["timestamp"], utc=True)
            .astype("int64")
            .to_numpy(dtype=float)
            / 1e9
        )

        u = np.full(n, np.nan)
        v = np.full(n, np.nan)
        speed = np.full(n, np.nan)
        direction = np.full(n, np.nan)
        distance = np.full(n, np.nan)
        dt_used = np.full(n, np.nan)

        def fill_velocity(target_idx, start_idx, end_idx, max_dt_s):
            if len(target_idx) == 0:
                return

            dt = t[end_idx] - t[start_idx]

            valid = (
                np.isfinite(lon[start_idx])
                & np.isfinite(lat[start_idx])
                & np.isfinite(lon[end_idx])
                & np.isfinite(lat[end_idx])
                & np.isfinite(dt)
                & (dt > 0)
                & (dt <= max_dt_s)
            )

            if not np.any(valid):
                return

            target_idx = target_idx[valid]
            start_idx = start_idx[valid]
            end_idx = end_idx[valid]
            dt = dt[valid]

            azimuth_deg, _, dist_m = self.geod.inv(
                lon[start_idx],
                lat[start_idx],
                lon[end_idx],
                lat[end_idx],
            )

            curr_speed = dist_m / dt

            # pyproj azimuth is degrees clockwise from north.
            azimuth_rad = np.deg2rad(azimuth_deg)

            curr_u = curr_speed * np.sin(azimuth_rad)
            curr_v = curr_speed * np.cos(azimuth_rad)

            u[target_idx] = curr_u
            v[target_idx] = curr_v
            speed[target_idx] = curr_speed
            direction[target_idx] = (azimuth_deg + 360.0) % 360.0
            distance[target_idx] = dist_m
            dt_used[target_idx] = dt

        idx = np.arange(n)

        # Central difference:
        # velocity at i is estimated from position i-1 to position i+1.
        # For hourly data, this is normally a centred 2-hour estimate.
        if n >= 3:
            target_idx = idx[1:-1]
            start_idx = idx[:-2]
            end_idx = idx[2:]

            gap_before = t[1:-1] - t[:-2]
            gap_after = t[2:] - t[1:-1]

            central_valid = (
                np.isfinite(gap_before)
                & np.isfinite(gap_after)
                & (gap_before > 0)
                & (gap_after > 0)
                & (gap_before <= max_gap_s)
                & (gap_after <= max_gap_s)
            )

            fill_velocity(
                target_idx=target_idx[central_valid],
                start_idx=start_idx[central_valid],
                end_idx=end_idx[central_valid],
                max_dt_s=2 * max_gap_s,
            )

        # One-sided fallback:
        # useful for first/last point and points near gaps.
        start_idx = idx[:-1]
        end_idx = idx[1:]

        pair_dt = t[end_idx] - t[start_idx]

        pair_valid = (
            np.isfinite(pair_dt)
            & (pair_dt > 0)
            & (pair_dt <= max_gap_s)
        )

        missing_forward = np.isnan(speed[:-1])
        valid_forward = pair_valid & missing_forward

        fill_velocity(
            target_idx=start_idx[valid_forward],
            start_idx=start_idx[valid_forward],
            end_idx=end_idx[valid_forward],
            max_dt_s=max_gap_s,
        )

        missing_backward = np.isnan(speed[1:])
        valid_backward = pair_valid & missing_backward

        fill_velocity(
            target_idx=end_idx[valid_backward],
            start_idx=start_idx[valid_backward],
            end_idx=end_idx[valid_backward],
            max_dt_s=max_gap_s,
        )

        qc_flag = np.full(n, 9, dtype=int)

        computed = np.isfinite(speed)
        qc_flag[computed] = 1

        too_fast = computed & (speed > max_speed_m_s)
        qc_flag[too_fast] = 4

        # Remove failed velocity values but keep diagnostics.
        u[too_fast] = np.nan
        v[too_fast] = np.nan
        speed[too_fast] = np.nan
        direction[too_fast] = np.nan

        group["drifter_u_m_s"] = u
        group["drifter_v_m_s"] = v
        group["drifter_speed_m_s"] = speed
        group["drifter_direction_to_deg"] = direction
        group["drifter_velocity_dt_s"] = dt_used
        group["drifter_velocity_distance_m"] = distance
        group["drifter_velocity_qc_flag"] = qc_flag

        return group

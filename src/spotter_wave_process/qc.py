from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


PASS = 1
NOT_EVALUATED = 2
SUSPECT = 3
FAIL = 4
MISSING = 9


@dataclass
class SpotterQCConfig:
    lat_min: float = -67.5
    lat_max: float = -50.5
    lon_min: float = -75.5
    lon_max: float = -30.5

    swh_suspect_min: float = 0.0
    swh_suspect_max: float = 15.0
    swh_fail_min: float = 0.0
    swh_fail_max: float = 25.0

    peak_period_suspect_min: float = 2.0
    peak_period_suspect_max: float = 25.0
    peak_period_fail_min: float = 1.0
    peak_period_fail_max: float = 35.0

    swh_spike_suspect_threshold: float = 2.0
    swh_spike_fail_threshold: float = 4.0

    swh_rate_suspect_threshold: float = 1.5
    swh_rate_fail_threshold: float = 3.0

    era5_swh_suspect_abs_error: float = 2.0
    era5_swh_fail_abs_error: float = 4.0

    era5_peak_period_suspect_abs_error: float = 4.0
    era5_peak_period_fail_abs_error: float = 8.0

    drifter_speed_suspect_ms: float = 1.5
    drifter_speed_fail_ms: float = 3.0

    stuck_distance_tolerance_m: float = 50.0
    stuck_suspect_duration_hours: float = 6.0
    stuck_fail_duration_hours: float = 12.0


@dataclass
class SpotterQARTODQC:
    df: pd.DataFrame
    config: SpotterQCConfig = field(default_factory=SpotterQCConfig)

    time_col: str = "timestamp"
    group_col: str = "spotter_id"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    swh_col: str = "significantWaveHeight"
    peak_period_col: str = "peakPeriod"
    era5_swh_error_col: str = "error_era5_swh"
    era5_peak_period_error_col: str = "error_era5_pp1d"

    def __post_init__(self) -> None:
        self.df = self.df.copy()

        if self.time_col not in self.df.columns:
            raise KeyError(f"Missing required time column: {self.time_col}")

        if self.group_col not in self.df.columns:
            raise KeyError(f"Missing required group column: {self.group_col}")

        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], utc=True)
        self.df = (
            self.df
            .sort_values([self.group_col, self.time_col])
            .reset_index(drop=True)
        )

    @staticmethod
    def haversine_m(lat1, lon1, lat2, lon2):
        r = 6_371_000.0

        lat1 = np.deg2rad(lat1)
        lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2)
        lon2 = np.deg2rad(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )

        return 2.0 * r * np.arcsin(np.sqrt(a))

    @staticmethod
    def gross_range(
        series: pd.Series,
        suspect_min: float,
        suspect_max: float,
        fail_min: float,
        fail_max: float,
    ) -> pd.Series:
        flag = pd.Series(PASS, index=series.index, dtype="int8")

        flag[series.isna()] = MISSING

        suspect = (
            ((series < suspect_min) | (series > suspect_max))
            & series.notna()
        )
        fail = (
            ((series < fail_min) | (series > fail_max))
            & series.notna()
        )

        flag[suspect] = SUSPECT
        flag[fail] = FAIL

        return flag

    @staticmethod
    def model_comparison(
        error: pd.Series,
        suspect_abs_error: float,
        fail_abs_error: float,
        fail_enabled: bool = True,
    ) -> pd.Series:
        flag = pd.Series(PASS, index=error.index, dtype="int8")
        abs_error = error.abs()

        flag[error.isna()] = NOT_EVALUATED
        flag[abs_error > suspect_abs_error] = SUSPECT

        if fail_enabled:
            flag[abs_error > fail_abs_error] = FAIL

        return flag

    def location_test(self) -> pd.Series:
        cfg = self.config
        df = self.df

        flag = pd.Series(PASS, index=df.index, dtype="int8")

        missing = df[self.lat_col].isna() | df[self.lon_col].isna()
        flag[missing] = MISSING

        outside = (
            (df[self.lat_col] < cfg.lat_min)
            | (df[self.lat_col] > cfg.lat_max)
            | (df[self.lon_col] < cfg.lon_min)
            | (df[self.lon_col] > cfg.lon_max)
        )

        flag[outside & ~missing] = FAIL

        self.df["location_qc"] = flag
        return flag

    def swh_gross_range_test(self) -> pd.Series:
        cfg = self.config

        flag = self.gross_range(
            self.df[self.swh_col],
            suspect_min=cfg.swh_suspect_min,
            suspect_max=cfg.swh_suspect_max,
            fail_min=cfg.swh_fail_min,
            fail_max=cfg.swh_fail_max,
        )

        self.df["swh_gross_range_qc"] = flag
        return flag

    def peak_period_gross_range_test(self) -> pd.Series:
        cfg = self.config

        flag = self.gross_range(
            self.df[self.peak_period_col],
            suspect_min=cfg.peak_period_suspect_min,
            suspect_max=cfg.peak_period_suspect_max,
            fail_min=cfg.peak_period_fail_min,
            fail_max=cfg.peak_period_fail_max,
        )

        self.df["peak_period_gross_range_qc"] = flag
        return flag

    def spike_test(
        self,
        column: str,
        output_column: str,
        suspect_threshold: float,
        fail_threshold: float,
    ) -> pd.Series:
        flag = pd.Series(PASS, index=self.df.index, dtype="int8")

        for _, g in self.df.groupby(self.group_col, sort=False):
            s = g[column]

            prev_v = s.shift(1)
            next_v = s.shift(-1)
            reference = 0.5 * (prev_v + next_v)

            spike = (s - reference).abs()

            local_flag = pd.Series(PASS, index=g.index, dtype="int8")
            local_flag[s.isna()] = MISSING
            local_flag[spike > suspect_threshold] = SUSPECT
            local_flag[spike > fail_threshold] = FAIL

            local_flag[prev_v.isna() | next_v.isna()] = NOT_EVALUATED
            local_flag[s.isna()] = MISSING

            flag.loc[g.index] = local_flag

        self.df[output_column] = flag
        return flag

    def swh_spike_test(self) -> pd.Series:
        cfg = self.config

        return self.spike_test(
            column=self.swh_col,
            output_column="swh_spike_qc",
            suspect_threshold=cfg.swh_spike_suspect_threshold,
            fail_threshold=cfg.swh_spike_fail_threshold,
        )

    def rate_of_change_test(
        self,
        column: str,
        output_column: str,
        suspect_rate: float,
        fail_rate: float,
    ) -> pd.Series:
        flag = pd.Series(PASS, index=self.df.index, dtype="int8")

        for _, g in self.df.groupby(self.group_col, sort=False):
            s = g[column]
            t = g[self.time_col]

            dt_hours = t.diff().dt.total_seconds() / 3600.0
            dv = s.diff().abs()
            rate = dv / dt_hours

            local_flag = pd.Series(PASS, index=g.index, dtype="int8")
            local_flag[s.isna()] = MISSING
            local_flag[dt_hours <= 0] = FAIL

            local_flag[rate > suspect_rate] = SUSPECT
            local_flag[rate > fail_rate] = FAIL

            local_flag[dt_hours.isna()] = NOT_EVALUATED
            local_flag[s.isna()] = MISSING

            flag.loc[g.index] = local_flag

        self.df[output_column] = flag
        return flag

    def swh_rate_of_change_test(self) -> pd.Series:
        cfg = self.config

        return self.rate_of_change_test(
            column=self.swh_col,
            output_column="swh_rate_of_change_qc",
            suspect_rate=cfg.swh_rate_suspect_threshold,
            fail_rate=cfg.swh_rate_fail_threshold,
        )

    def era5_swh_comparison_test(self, fail_enabled: bool = True) -> pd.Series:
        cfg = self.config

        flag = self.model_comparison(
            self.df[self.era5_swh_error_col],
            suspect_abs_error=cfg.era5_swh_suspect_abs_error,
            fail_abs_error=cfg.era5_swh_fail_abs_error,
            fail_enabled=fail_enabled,
        )

        self.df["era5_swh_comparison_qc"] = flag
        return flag

    def era5_peak_period_comparison_test(
        self,
        fail_enabled: bool = True,
    ) -> pd.Series:
        cfg = self.config

        flag = self.model_comparison(
            self.df[self.era5_peak_period_error_col],
            suspect_abs_error=cfg.era5_peak_period_suspect_abs_error,
            fail_abs_error=cfg.era5_peak_period_fail_abs_error,
            fail_enabled=fail_enabled,
        )

        self.df["era5_peak_period_comparison_qc"] = flag
        return flag

    def drifter_speed_test(self) -> pd.Series:
        cfg = self.config
        flag = pd.Series(PASS, index=self.df.index, dtype="int8")

        for _, g in self.df.groupby(self.group_col, sort=False):
            lat = g[self.lat_col]
            lon = g[self.lon_col]
            t = g[self.time_col]

            distance = self.haversine_m(
                lat.shift(1),
                lon.shift(1),
                lat,
                lon,
            )

            dt = t.diff().dt.total_seconds()
            speed = distance / dt

            local_flag = pd.Series(PASS, index=g.index, dtype="int8")

            missing = lat.isna() | lon.isna() | t.isna()
            local_flag[missing] = MISSING

            local_flag[dt <= 0] = FAIL
            local_flag[speed > cfg.drifter_speed_suspect_ms] = SUSPECT
            local_flag[speed > cfg.drifter_speed_fail_ms] = FAIL

            local_flag[dt.isna()] = NOT_EVALUATED
            local_flag[missing] = MISSING

            flag.loc[g.index] = local_flag

        self.df["drifter_speed_qc"] = flag
        return flag

    def stuck_position_test(self) -> pd.Series:
        cfg = self.config
        flag = pd.Series(PASS, index=self.df.index, dtype="int8")

        for _, g in self.df.groupby(self.group_col, sort=False):
            lat = g[self.lat_col]
            lon = g[self.lon_col]
            t = g[self.time_col]

            step_distance = self.haversine_m(
                lat.shift(1),
                lon.shift(1),
                lat,
                lon,
            )

            dt_hours = t.diff().dt.total_seconds() / 3600.0

            small_movement = step_distance <= cfg.stuck_distance_tolerance_m

            local_flag = pd.Series(PASS, index=g.index, dtype="int8")

            missing = lat.isna() | lon.isna() | t.isna()
            local_flag[missing] = MISSING

            run_id = (small_movement != small_movement.shift()).cumsum()

            for _, run_idx in g.groupby(run_id).groups.items():
                run_idx = list(run_idx)

                if not bool(small_movement.loc[run_idx].all()):
                    continue

                duration = dt_hours.loc[run_idx].sum(skipna=True)

                if duration >= cfg.stuck_fail_duration_hours:
                    local_flag.loc[run_idx] = FAIL
                elif duration >= cfg.stuck_suspect_duration_hours:
                    local_flag.loc[run_idx] = SUSPECT

            if len(local_flag) > 0:
                local_flag.iloc[0] = NOT_EVALUATED

            local_flag[missing] = MISSING

            flag.loc[g.index] = local_flag

        self.df["stuck_position_qc"] = flag
        return flag

    @staticmethod
    def summary_flag(flag_columns: pd.DataFrame) -> pd.Series:
        summary = pd.Series(PASS, index=flag_columns.index, dtype="int8")

        any_fail = (flag_columns == FAIL).any(axis=1)
        any_suspect = (flag_columns == SUSPECT).any(axis=1)
        all_missing = (flag_columns == MISSING).all(axis=1)
        all_not_eval = (flag_columns == NOT_EVALUATED).all(axis=1)

        summary[all_not_eval] = NOT_EVALUATED
        summary[any_suspect] = SUSPECT
        summary[any_fail] = FAIL
        summary[all_missing] = MISSING

        return summary

    def build_summary_flag(
        self,
        qc_columns: Iterable[str] | None = None,
    ) -> pd.Series:
        if qc_columns is None:
            qc_columns = self.qc_columns

        qc_columns = list(qc_columns)

        self.df["summary_qc"] = self.summary_flag(self.df[qc_columns])
        return self.df["summary_qc"]

    @property
    def qc_columns(self) -> list[str]:
        return [col for col in self.df.columns if col.endswith("_qc") and col != "summary_qc"]

    def run_all(
        self,
        model_comparison_can_fail: bool = False,
        remove_failed_after: bool = False,
    ) -> pd.DataFrame:
        """
        Run all QC tests.

        By default, ERA5 comparison can mark data as suspect but not fail.
        This avoids rejecting good observations only because the model differs.
        """
        self.location_test()
        self.swh_gross_range_test()
        self.peak_period_gross_range_test()
        self.swh_spike_test()
        self.swh_rate_of_change_test()

        self.era5_swh_comparison_test(
            fail_enabled=model_comparison_can_fail,
        )
        self.era5_peak_period_comparison_test(
            fail_enabled=model_comparison_can_fail,
        )

        self.drifter_speed_test()
        self.stuck_position_test()

        self.build_summary_flag()

        if remove_failed_after:
            self.remove_failed(inplace=True)
        return self.df

    def flag_counts(self) -> pd.DataFrame:
        records = []

        flag_names = {
            PASS: "pass",
            NOT_EVALUATED: "not_evaluated",
            SUSPECT: "suspect",
            FAIL: "fail",
            MISSING: "missing",
        }

        for col in self.qc_columns + ["summary_qc"]:
            if col not in self.df:
                continue

            counts = self.df[col].value_counts(dropna=False).sort_index()

            for flag, count in counts.items():
                records.append(
                    {
                        "test": col,
                        "flag": int(flag),
                        "meaning": flag_names.get(int(flag), "unknown"),
                        "count": int(count),
                    }
                )

        return pd.DataFrame.from_records(records)

    def failed_or_suspect(self) -> pd.DataFrame:
        if "summary_qc" not in self.df:
            raise ValueError("Run QC first with .run_all().")

        return self.df[self.df["summary_qc"].isin([SUSPECT, FAIL])].copy()

    def remove_failed(self, qc_column: str = "summary_qc", inplace: bool = True) -> pd.DataFrame:
        """
        Remove rows where the selected QC column is FAIL.

        By default, removes rows where summary_qc == 4.
        """
        if qc_column not in self.df.columns:
            raise ValueError(f"QC column {qc_column!r} not found. Run QC first.")

        filtered = self.df[self.df[qc_column] != FAIL].copy()

        if inplace:
            self.df = filtered.reset_index(drop=True)
            return self.df

        return filtered.reset_index(drop=True)

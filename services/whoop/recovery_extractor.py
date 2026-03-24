import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Any

from .client import WhoopApiClient, WhoopConfiguration, WhoopIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class WhoopRecoverySnapshot:
    user_profile: dict[str, Any] | None
    body_measurements: dict[str, Any] | None
    recovery_indicators: list[dict[str, Any]]
    latest_resting_heart_rate: int | None = None
    latest_hrv_rmssd_milli: float | None = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _hours_from_millis(value: Any) -> float | None:
    millis = _safe_float(value)
    return round(millis / 3_600_000, 2) if millis is not None else None


def _timezone_from_offset(offset: str | None) -> timezone:
    raw = str(offset or "+00:00").strip()
    if len(raw) != 6 or raw[0] not in {"+", "-"} or raw[3] != ":":
        return timezone.utc

    sign = 1 if raw[0] == "+" else -1
    hours = int(raw[1:3])
    minutes = int(raw[4:6])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _local_date(dt: datetime | None, offset: str | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_timezone_from_offset(offset)).date().isoformat()


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else None
    return None


def _getattr_path(container: Any, *parts: str) -> Any:
    cur = container
    for part in parts:
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _map_sleep(sleep: Any, recovery: Any) -> dict[str, Any] | None:
    if sleep is None:
        return None

    stage_summary = _getattr_path(sleep, "score", "stage_summary")
    sleep_needed = _getattr_path(sleep, "score", "sleep_needed")
    sleep_score = getattr(sleep, "score", None)
    recovery_score = getattr(recovery, "score", None)

    return {
        "duration": {
            "total": _hours_from_millis(_getattr_path(stage_summary, "total_sleep_time_milli")),
            "deep": _hours_from_millis(_getattr_path(stage_summary, "total_slow_wave_sleep_time_milli")),
            "light": _hours_from_millis(_getattr_path(stage_summary, "total_light_sleep_time_milli")),
            "rem": _hours_from_millis(_getattr_path(stage_summary, "total_rem_sleep_time_milli")),
            "awake": _hours_from_millis(_getattr_path(stage_summary, "total_awake_time_milli")),
        },
        "quality": {
            "overall_score": _safe_float(_getattr_path(sleep_score, "sleep_performance_percentage")),
            "deep_sleep": None,
            "rem_sleep": None,
        },
        "restless_moments": _safe_int(_getattr_path(stage_summary, "disturbance_count")),
        "avg_overnight_hrv": _safe_float(_getattr_path(recovery_score, "hrv_rmssd_milli")),
        "resting_heart_rate": _safe_int(_getattr_path(recovery_score, "resting_heart_rate")),
        "sleep_consistency_percentage": _safe_float(_getattr_path(sleep_score, "sleep_consistency_percentage")),
        "sleep_efficiency_percentage": _safe_float(_getattr_path(sleep_score, "sleep_efficiency_percentage")),
        "sleep_needed_hours": _hours_from_millis(_getattr_path(sleep_needed, "total_need_milli")),
        "respiratory_rate": _safe_float(_getattr_path(sleep_score, "respiratory_rate")),
    }


def _map_recovery(recovery: Any) -> dict[str, Any] | None:
    if recovery is None:
        return None

    score = getattr(recovery, "score", None)
    return {
        "recovery_score": _safe_float(_getattr_path(score, "recovery_score")),
        "resting_heart_rate": _safe_int(_getattr_path(score, "resting_heart_rate")),
        "hrv_rmssd_milli": _safe_float(_getattr_path(score, "hrv_rmssd_milli")),
        "spo2_percentage": _safe_float(_getattr_path(score, "spo2_percentage")),
        "skin_temp_celsius": _safe_float(_getattr_path(score, "skin_temp_celsius")),
        "user_calibrating": bool(_getattr_path(score, "user_calibrating")),
    }


class WhoopRecoveryExtractor:
    def __init__(self, config: WhoopConfiguration):
        self.config = config

    def extract_data(self, start_date: date, end_date: date) -> WhoopRecoverySnapshot:
        client = WhoopApiClient(self.config)
        client.connect()

        try:
            user_profile = self._try_best_effort(client.get_profile, "profile")
            body_measurements = self._try_best_effort(client.get_body_measurements, "body measurements")

            start_dt = datetime.combine(start_date, time.min, tzinfo=UTC)
            end_dt = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)

            cycles = client.get_cycles(start=start_dt, end=end_dt)
            recoveries = client.get_recoveries(start=start_dt, end=end_dt)
            sleeps = client.get_sleeps(start=start_dt, end=end_dt)

            sleeps_by_id = {
                str(getattr(sleep, "id", "")): sleep
                for sleep in sleeps
                if getattr(sleep, "id", None) is not None
            }
            sleeps_by_date = {
                anchored: sleep
                for sleep in sleeps
                if (anchored := _local_date(getattr(sleep, "end", None), getattr(sleep, "timezone_offset", None)))
            }
            recoveries_by_cycle_id = {
                int(getattr(recovery, "cycle_id")): recovery
                for recovery in recoveries
                if getattr(recovery, "cycle_id", None) is not None
            }

            indicators: list[dict[str, Any]] = []
            latest_marker_date = ""
            latest_resting_heart_rate: int | None = None
            latest_hrv: float | None = None

            for cycle in cycles:
                anchored_date = _local_date(
                    getattr(cycle, "end", None) or getattr(cycle, "start", None),
                    getattr(cycle, "timezone_offset", None),
                )
                if anchored_date is None:
                    continue

                recovery = recoveries_by_cycle_id.get(int(getattr(cycle, "id")))
                sleep = None
                if recovery is not None and getattr(recovery, "sleep_id", None) is not None:
                    sleep = sleeps_by_id.get(str(getattr(recovery, "sleep_id")))
                if sleep is None:
                    sleep = sleeps_by_date.get(anchored_date)

                mapped_recovery = _map_recovery(recovery)
                indicators.append(
                    {
                        "date": anchored_date,
                        "sleep": _map_sleep(sleep, recovery),
                        "recovery": mapped_recovery,
                        "day_strain": _safe_float(_getattr_path(cycle, "score", "strain")),
                    }
                )

                if mapped_recovery and anchored_date >= latest_marker_date:
                    latest_marker_date = anchored_date
                    latest_resting_heart_rate = _safe_int(mapped_recovery.get("resting_heart_rate"))
                    latest_hrv = _safe_float(mapped_recovery.get("hrv_rmssd_milli"))

            return WhoopRecoverySnapshot(
                user_profile=_model_dump(user_profile),
                body_measurements=_model_dump(body_measurements),
                recovery_indicators=sorted(indicators, key=lambda item: item["date"]),
                latest_resting_heart_rate=latest_resting_heart_rate,
                latest_hrv_rmssd_milli=latest_hrv,
            )
        except WhoopIntegrationError:
            raise
        except Exception as exc:
            raise WhoopIntegrationError(f"WHOOP recovery extraction failed: {exc}") from exc
        finally:
            client.close()

    @staticmethod
    def _try_best_effort(fn: Any, label: str) -> Any:
        try:
            return fn()
        except Exception:
            logger.warning("WHOOP %s fetch failed; continuing without it", label, exc_info=True)
            return None

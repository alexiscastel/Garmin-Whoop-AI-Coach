from copy import deepcopy
from typing import Any

from .recovery_extractor import WhoopRecoverySnapshot


def _merge_non_null(base: Any, overlay: Any) -> Any:
    if overlay is None:
        return deepcopy(base)
    if base is None:
        return deepcopy(overlay)
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return deepcopy(overlay)

    merged = deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        existing = merged.get(key)
        merged[key] = _merge_non_null(existing, value)
    return merged


def merge_whoop_into_garmin_data(
    garmin_data: dict[str, Any],
    whoop_data: WhoopRecoverySnapshot,
) -> dict[str, Any]:
    merged = deepcopy(garmin_data)

    recovery_indicators = list(merged.get("recovery_indicators") or [])
    by_date = {
        item.get("date"): item
        for item in recovery_indicators
        if isinstance(item, dict) and item.get("date")
    }

    for item in whoop_data.recovery_indicators:
        date_key = item.get("date")
        if not date_key:
            continue

        target = by_date.get(date_key)
        if target is None:
            target = {
                "date": date_key,
                "sleep": None,
                "stress": None,
                "recovery": None,
                "day_strain": None,
            }
            recovery_indicators.append(target)
            by_date[date_key] = target

        if item.get("sleep"):
            target["sleep"] = _merge_non_null(target.get("sleep"), item["sleep"])
        if item.get("recovery"):
            target["recovery"] = _merge_non_null(target.get("recovery"), item["recovery"])
        if item.get("day_strain") is not None:
            target["day_strain"] = float(item["day_strain"])

    if recovery_indicators:
        merged["recovery_indicators"] = sorted(
            recovery_indicators,
            key=lambda entry: str(entry.get("date", "")),
        )

    physiological_markers = dict(merged.get("physiological_markers") or {})
    if whoop_data.latest_resting_heart_rate is not None:
        physiological_markers["resting_heart_rate"] = whoop_data.latest_resting_heart_rate

    hrv = dict(physiological_markers.get("hrv") or {})
    if whoop_data.latest_hrv_rmssd_milli is not None:
        hrv["last_night_avg"] = whoop_data.latest_hrv_rmssd_milli
    if hrv:
        physiological_markers["hrv"] = hrv
    if physiological_markers:
        merged["physiological_markers"] = physiological_markers

    return merged

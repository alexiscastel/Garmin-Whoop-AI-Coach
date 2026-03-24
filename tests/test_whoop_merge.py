from services.whoop.merge import merge_whoop_into_garmin_data
from services.whoop.recovery_extractor import WhoopRecoverySnapshot


def test_merge_whoop_overrides_recovery_and_preserves_garmin_stress():
    garmin_data = {
        "recent_activities": [{"activity_id": 1}],
        "training_load_history": [{"date": "2024-01-01", "load": 500}],
        "physiological_markers": {
            "resting_heart_rate": 52,
            "vo2_max": 58.0,
            "hrv": {"weekly_avg": 61.0, "last_night_avg": 55.0},
        },
        "recovery_indicators": [
            {
                "date": "2024-01-01",
                "sleep": {
                    "duration": {"total": 7.0, "deep": 1.5},
                    "quality": {"overall_score": 70},
                    "resting_heart_rate": 52,
                },
                "stress": {"avg_level": 24, "max_level": 55},
            }
        ],
    }
    whoop_data = WhoopRecoverySnapshot(
        user_profile=None,
        body_measurements=None,
        recovery_indicators=[
            {
                "date": "2024-01-01",
                "sleep": {
                    "duration": {"total": 8.1},
                    "quality": {"overall_score": 92},
                    "resting_heart_rate": 48,
                    "avg_overnight_hrv": 63.5,
                },
                "recovery": {"recovery_score": 88.0, "resting_heart_rate": 48, "hrv_rmssd_milli": 63.5},
                "day_strain": 11.4,
            }
        ],
        latest_resting_heart_rate=48,
        latest_hrv_rmssd_milli=63.5,
    )

    merged = merge_whoop_into_garmin_data(garmin_data, whoop_data)

    indicator = merged["recovery_indicators"][0]
    assert indicator["sleep"]["duration"]["total"] == 8.1
    assert indicator["sleep"]["duration"]["deep"] == 1.5
    assert indicator["sleep"]["quality"]["overall_score"] == 92
    assert indicator["stress"] == {"avg_level": 24, "max_level": 55}
    assert indicator["recovery"]["recovery_score"] == 88.0
    assert indicator["day_strain"] == 11.4
    assert merged["recent_activities"] == [{"activity_id": 1}]
    assert merged["training_load_history"] == [{"date": "2024-01-01", "load": 500}]
    assert merged["physiological_markers"]["resting_heart_rate"] == 48
    assert merged["physiological_markers"]["vo2_max"] == 58.0
    assert merged["physiological_markers"]["hrv"]["weekly_avg"] == 61.0
    assert merged["physiological_markers"]["hrv"]["last_night_avg"] == 63.5


def test_merge_whoop_adds_new_recovery_day_without_touching_training_load():
    garmin_data = {"training_load_history": [{"date": "2024-01-01", "load": 500}]}
    whoop_data = WhoopRecoverySnapshot(
        user_profile=None,
        body_measurements=None,
        recovery_indicators=[
            {
                "date": "2024-01-02",
                "sleep": {"duration": {"total": 7.8}},
                "recovery": {"recovery_score": 80.0},
                "day_strain": 10.0,
            }
        ],
        latest_resting_heart_rate=None,
        latest_hrv_rmssd_milli=None,
    )

    merged = merge_whoop_into_garmin_data(garmin_data, whoop_data)

    assert merged["training_load_history"] == [{"date": "2024-01-01", "load": 500}]
    assert merged["recovery_indicators"][0]["date"] == "2024-01-02"
    assert merged["recovery_indicators"][0]["day_strain"] == 10.0

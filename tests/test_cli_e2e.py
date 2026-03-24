import json
from unittest.mock import AsyncMock, patch

import pytest

from services.garmin.models import GarminData
from services.whoop.recovery_extractor import WhoopRecoverySnapshot


@pytest.mark.asyncio
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
async def test_cli_e2e_smoke_with_mocks(
    mock_outside_client,
    mock_extractor_class,
    tmp_path,
):
    """Test CLI end-to-end with all external dependencies mocked."""
    # Configure workflow mock
    workflow_result = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec",
        "execution_metadata": {"trace_id": "trace-1", "root_run_id": "root-1"},
    }

    # Configure extractor mock
    mock_instance = mock_extractor_class.return_value
    mock_instance.extract_data.return_value = GarminData()

    # Configure outside client mock
    mock_outside_instance = mock_outside_client.return_value
    mock_outside_instance.get_competitions.return_value = []

    # Import after patches are in place
    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test A"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    with patch("cli.garmin_ai_coach_cli._run_complete_analysis_and_planning", new=AsyncMock(return_value=workflow_result)):
        await run_analysis_from_config(config_path)

    analysis_path = output_directory / "analysis.html"
    planning_path = output_directory / "planning.html"
    summary_path = output_directory / "summary.json"
    assert analysis_path.exists()
    assert planning_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["athlete"] == "Test A"
    assert summary["total_cost_usd"] == 0.0
    assert summary["data_sources"] == ["garmin"]
    assert summary["whoop_enabled"] is False
    assert summary["whoop_status"] == "disabled"


@pytest.mark.asyncio
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
@patch("getpass.getpass", return_value="dummy")
@patch("builtins.input", side_effect=["My goal is to complete a marathon"])
async def test_cli_e2e_with_hitl_enabled(
    mock_input,
    mock_getpass,
    mock_outside_client,
    mock_extractor_class,
    tmp_path,
):
    """Test CLI with HITL enabled to ensure user interactions work."""
    # Configure workflow mock
    workflow_result = {
        "analysis_html": "<html><body>Analysis with HITL</body></html>",
        "planning_html": "<html><body>Plan with HITL</body></html>",
       "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.05, "total_tokens": 1000},
        "execution_id": "test-exec-hitl",
        "execution_metadata": {"trace_id": "trace-hitl", "root_run_id": "root-hitl"},
    }

    # Configure extractor mock
    mock_instance = mock_extractor_class.return_value
    mock_instance.extract_data.return_value = GarminData()

    # Configure outside client mock
    mock_outside_instance = mock_outside_client.return_value
    mock_outside_instance.get_competitions.return_value = []

    # Import after patches are in place
    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out_hitl"
    config_path = tmp_path / "config_hitl.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Athlete HITL"
  email: "user@example.com"

context:
  analysis: "HITL Analysis context"
  planning: "HITL Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: true

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    with patch("cli.garmin_ai_coach_cli._run_complete_analysis_and_planning", new=AsyncMock(return_value=workflow_result)):
        await run_analysis_from_config(config_path)

    analysis_path = output_directory / "analysis.html"
    planning_path = output_directory / "planning.html"
    summary_path = output_directory / "summary.json"

    assert analysis_path.exists()
    assert planning_path.exists()
    assert summary_path.exists()

    # Verify the basic structure is correct
    assert analysis_path.read_text(encoding="utf-8").startswith("<html>")
    assert planning_path.read_text(encoding="utf-8").startswith("<html>")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["athlete"] == "Test Athlete HITL"
    assert "total_cost_usd" in summary
    assert "total_tokens" in summary
    assert summary["data_sources"] == ["garmin"]


@pytest.mark.asyncio
@patch("cli.garmin_ai_coach_cli.WhoopRecoveryExtractor")
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
async def test_cli_e2e_with_whoop_success(
    mock_outside_client,
    mock_extractor_class,
    mock_whoop_extractor_class,
    tmp_path,
):
    workflow_result = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec",
        "execution_metadata": {"trace_id": "trace-1", "root_run_id": "root-1"},
    }

    mock_extractor_class.return_value.extract_data.return_value = GarminData()
    mock_outside_client.return_value.get_competitions.return_value = []
    mock_whoop_extractor_class.return_value.extract_data.return_value = WhoopRecoverySnapshot(
        user_profile=None,
        body_measurements=None,
        recovery_indicators=[
            {
                "date": "2024-01-01",
                "sleep": {"duration": {"total": 8.0}},
                "recovery": {"recovery_score": 82.0},
                "day_strain": 11.2,
            }
        ],
        latest_resting_heart_rate=48,
        latest_hrv_rmssd_milli=62.5,
    )

    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out_whoop"
    config_path = tmp_path / "config_whoop.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Whoop"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false

whoop:
  enabled: true
  client_id: "client-id"
  client_secret: "client-secret"

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    with patch("cli.garmin_ai_coach_cli._run_complete_analysis_and_planning", new=AsyncMock(return_value=workflow_result)):
        await run_analysis_from_config(config_path)

    summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_sources"] == ["garmin", "whoop"]
    assert summary["whoop_enabled"] is True
    assert summary["whoop_status"] == "success"


@pytest.mark.asyncio
@patch("cli.garmin_ai_coach_cli.WhoopRecoveryExtractor")
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
@patch("builtins.input", return_value="yes")
async def test_cli_e2e_with_whoop_fallback_after_prompt(
    mock_input,
    mock_outside_client,
    mock_extractor_class,
    mock_whoop_extractor_class,
    tmp_path,
    monkeypatch,
):
    workflow_result = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec",
        "execution_metadata": {"trace_id": "trace-1", "root_run_id": "root-1"},
    }

    mock_extractor_class.return_value.extract_data.return_value = GarminData()
    mock_outside_client.return_value.get_competitions.return_value = []
    mock_whoop_extractor_class.return_value.extract_data.side_effect = RuntimeError("auth failed")

    from cli.garmin_ai_coach_cli import run_analysis_from_config

    monkeypatch.setattr("cli.garmin_ai_coach_cli._can_prompt_user", lambda: True)

    output_directory = tmp_path / "out_whoop_fallback"
    config_path = tmp_path / "config_whoop_fallback.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Whoop Fallback"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false

whoop:
  enabled: true
  client_id: "client-id"
  client_secret: "client-secret"

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    with patch("cli.garmin_ai_coach_cli._run_complete_analysis_and_planning", new=AsyncMock(return_value=workflow_result)):
        await run_analysis_from_config(config_path)

    summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_sources"] == ["garmin"]
    assert summary["whoop_enabled"] is True
    assert summary["whoop_status"] == "fallback_to_garmin"


@pytest.mark.asyncio
@patch("cli.garmin_ai_coach_cli.WhoopRecoveryExtractor")
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
async def test_cli_e2e_with_whoop_fallback_non_interactive(
    mock_outside_client,
    mock_extractor_class,
    mock_whoop_extractor_class,
    tmp_path,
    monkeypatch,
):
    workflow_result = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec",
        "execution_metadata": {"trace_id": "trace-1", "root_run_id": "root-1"},
    }

    mock_extractor_class.return_value.extract_data.return_value = GarminData()
    mock_outside_client.return_value.get_competitions.return_value = []
    mock_whoop_extractor_class.return_value.extract_data.side_effect = RuntimeError("api down")

    from cli.garmin_ai_coach_cli import run_analysis_from_config

    monkeypatch.setattr("cli.garmin_ai_coach_cli._can_prompt_user", lambda: False)

    output_directory = tmp_path / "out_whoop_noninteractive"
    config_path = tmp_path / "config_whoop_noninteractive.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Whoop NonInteractive"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false

whoop:
  enabled: true
  client_id: "client-id"
  client_secret: "client-secret"

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    with patch("cli.garmin_ai_coach_cli._run_complete_analysis_and_planning", new=AsyncMock(return_value=workflow_result)):
        await run_analysis_from_config(config_path)

    summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_sources"] == ["garmin"]
    assert summary["whoop_enabled"] is True
    assert summary["whoop_status"] == "fallback_to_garmin"

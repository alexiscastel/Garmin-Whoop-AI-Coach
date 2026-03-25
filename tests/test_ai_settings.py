from core.config import AIMode
from services.ai.ai_settings import AISettings, AgentRole


def test_cost_effective_mode_uses_current_haiku_model():
    settings = AISettings(mode=AIMode.COST_EFFECTIVE)

    assert settings.get_model_for_role(AgentRole.SUMMARIZER) == "claude-haiku-4.5"
    assert settings.get_model_for_role(AgentRole.METRICS_EXPERT) == "claude-haiku-4.5"
    assert settings.get_model_for_role(AgentRole.SEASON_PLANNER) == "claude-haiku-4.5"

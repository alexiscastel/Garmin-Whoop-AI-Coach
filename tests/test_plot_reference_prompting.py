import pytest

from services.ai.langgraph.nodes.prompt_components import (
    get_available_plot_references_context,
    get_plotting_instructions,
)
from services.ai.langgraph.nodes.synthesis_node import synthesis_node


def test_plotting_instructions_require_tool_returned_ids():
    instructions = get_plotting_instructions("metrics")

    assert "exact returned tag `[PLOT:<plot_id>]`" in instructions
    assert "Do NOT make up semantic or placeholder IDs" in instructions
    assert "[PLOT:training_load_overview]" in instructions


def test_available_plot_references_context_lists_only_real_ids():
    context = get_available_plot_references_context([
        "metrics_111_001",
        "metrics_111_001",
        "activity_222_001",
    ])

    assert "Use ONLY these exact plot IDs" in context
    assert context.count("metrics_111_001") == 1
    assert "`activity_222_001`" in context
    assert "[PLOT:training_load_overview]" in context


@pytest.mark.asyncio
async def test_synthesis_node_includes_available_plot_ids_in_prompt(monkeypatch):
    captured = {}

    class DummyLLM:
        def bind_tools(self, tools):
            return self

    async def fake_handle_tool_calling_in_node(llm_with_tools, messages, tools, max_iterations):
        captured["messages"] = messages
        return "Synthesized report"

    async def fake_retry_with_backoff(func, *_args):
        return await func()

    monkeypatch.setattr(
        "services.ai.langgraph.nodes.synthesis_node.handle_tool_calling_in_node",
        fake_handle_tool_calling_in_node,
    )
    monkeypatch.setattr(
        "services.ai.langgraph.nodes.synthesis_node.retry_with_backoff",
        fake_retry_with_backoff,
    )
    monkeypatch.setattr(
        "services.ai.langgraph.nodes.synthesis_node.extract_expert_output",
        lambda *_args, **_kwargs: "Expert analysis with [PLOT:metrics_123_001]",
    )
    monkeypatch.setattr(
        "services.ai.langgraph.nodes.synthesis_node.ModelSelector.get_llm",
        lambda _role: DummyLLM(),
    )

    state = {
        "execution_id": "exec_1",
        "athlete_name": "Alexis",
        "competitions": [],
        "current_date": {},
        "style_guide": "",
        "plotting_enabled": True,
        "available_plots": ["metrics_123_001", "metrics_123_001", "activity_456_001"],
        "metrics_outputs": {},
        "activity_outputs": {},
        "physiology_outputs": {},
    }

    result = await synthesis_node(state)

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]

    assert "`metrics_123_001`" in system_prompt
    assert "`activity_456_001`" in system_prompt
    assert "[PLOT:training_load_overview]" in system_prompt
    assert "`metrics_123_001`" in user_prompt
    assert result["available_plots"] == ["activity_456_001", "metrics_123_001"]

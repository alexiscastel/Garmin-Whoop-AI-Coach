import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.output_helper import extract_expert_output
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .prompt_components import get_available_plot_references_context
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)





SYNTHESIS_SYSTEM_PROMPT_BASE = """You are a performance integration specialist.
## Goal
Create comprehensive, actionable insights by synthesizing multiple data streams.
## Principles
- Integrate: Connect insights from metrics, activity, and physiology.
- Contextualize: Relate data to the athlete's history and goals.
- Simplify: Make complex relationships understandable."""

SYNTHESIS_PLOT_INSTRUCTIONS = """
## Plot Integration
- Include plot references as `[PLOT:plot_id]` in your text.
- Use only real plot IDs that already exist.
- Never invent placeholder IDs such as `[PLOT:training_load_overview]`.
- These will become interactive charts."""

SYNTHESIS_USER_PROMPT_BASE = """Synthesize the expert analyses into a comprehensive athlete report.

## Inputs
### Metrics
```markdown
{metrics_result}
```
### Activity
```markdown
{activity_result}
```
### Physiology
```markdown
{physiology_result}
```
### Context
- Competitions: ```json {competitions} ```
- Date: ```json {current_date} ```
- Style: ```markdown {style_guide} ```

## Task
1. **Integrate**: Connect load (metrics), execution (activity), and response (physiology).
2. **Identify Patterns**: Spot trends in performance and adaptation.
3. **Synthesize**: Create a coherent story, not just a list of facts.

## Output Format
- **Executive Summary**: High-level status and key takeaways.
- **Key Performance Indicators**: Table format.
- **Deep Dive**: Structured sections with clear headings.
- **Recommendations**: Brief and actionable.
- **Tone**: Professional, evidence-based, encouraging."""

SYNTHESIS_USER_PLOT_INSTRUCTIONS = """
## Plot References
- Include each unique `[PLOT:plot_id]` EXACTLY ONCE.
- Do not duplicate references."""


async def synthesis_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting synthesis node")

    try:
        plotting_enabled = state.get("plotting_enabled", False)
        available_plot_ids = sorted(set(state.get("available_plots", [])))
        plot_reference_context = (
            get_available_plot_references_context(available_plot_ids)
            if plotting_enabled else ""
        )

        logger.info(
            "Synthesis node: Plotting %s - %s plot integration instructions (%s available plots)",
            "enabled" if plotting_enabled else "disabled",
            "including" if plotting_enabled else "no",
            len(available_plot_ids),
        )

        agent_start_time = datetime.now()

        async def call_synthesis_analysis():
            return await handle_tool_calling_in_node(
                llm_with_tools=ModelSelector.get_llm(AgentRole.SYNTHESIS).bind_tools([]),
                messages=[
                    {"role": "system", "content": (
                        SYNTHESIS_SYSTEM_PROMPT_BASE
                        + (SYNTHESIS_PLOT_INSTRUCTIONS if plotting_enabled else "")
                        + plot_reference_context
                    )},
                    {"role": "user", "content": (
                        SYNTHESIS_USER_PROMPT_BASE.format(
                            athlete_name=state["athlete_name"],
                            metrics_result=extract_expert_output(state.get("metrics_outputs"), "for_synthesis"),
                            activity_result=extract_expert_output(state.get("activity_outputs"), "for_synthesis"),
                            physiology_result=extract_expert_output(state.get("physiology_outputs"), "for_synthesis"),
                            competitions=json.dumps(state["competitions"], indent=2),
                            current_date=json.dumps(state["current_date"], indent=2),
                            style_guide=state["style_guide"],
                        )
                        + (SYNTHESIS_USER_PLOT_INSTRUCTIONS if plotting_enabled else "")
                        + plot_reference_context
                    )},
                ],
                tools=[],
                max_iterations=3,
            )

        synthesis_result = await retry_with_backoff(
            call_synthesis_analysis, AI_ANALYSIS_CONFIG, "Synthesis Analysis with Tools"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Synthesis analysis completed in %.2fs", execution_time)

        return {
            "synthesis_result": synthesis_result,
            "synthesis_complete": True,
            "costs": [
                {
                    "agent": "synthesis",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "available_plots": available_plot_ids,
        }

    except Exception as exc:
        logger.exception("Synthesis node failed")
        return {"errors": [f"Synthesis analysis failed: {exc!s}"]}

from typing import Literal

AgentType = Literal[
    "metrics_summarizer",
    "physiology_summarizer",
    "activity_summarizer",
    "metrics",
    "physiology",
    "activity",
    "synthesis",
    "season_planner",
    "weekly_planner",
]


def get_workflow_context(agent_type: AgentType) -> str:
    # Summarizer agents
    if agent_type in ["metrics_summarizer", "physiology_summarizer", "activity_summarizer"]:
        domain = agent_type.replace("_summarizer", "")
        return f"""
## System Role
You are the **{agent_type.replace('_', ' ').title()}**.
- **Input**: Raw `garmin_data`
- **Output**: Structured `{domain}_summary`
- **Goal**: Condense raw data into a factual, structured summary for the {domain} expert. Do NOT interpret."""

    # Expert agents
    if agent_type in ["metrics", "physiology", "activity"]:
        return f"""
## System Role
You are the **{agent_type.title()} Expert**.
- **Input**: `{agent_type}_summary`
- **Output**: `{agent_type}_outputs` with 3 fields:
  1. `for_synthesis`: For the comprehensive report.
  2. `for_season_planner`: Strategic insights (12-24 weeks).
  3. `for_weekly_planner`: Tactical details (next 28 days).
- **Goal**: Analyze patterns and provide specific insights for each consumer.
- **Context**: You are 1 of 3 parallel experts. Focus ONLY on your domain."""

    # Synthesis agent
    if agent_type == "synthesis":
        return """
## System Role
You are the **Synthesis Agent**.
- **Input**: `for_synthesis` fields from Metrics, Physiology, and Activity experts.
- **Output**: `synthesis_result` (Comprehensive Athlete Report).
- **Goal**: Integrate domain insights into a coherent story. Focus on historical patterns, not future planning."""

    # Planner agents
    if agent_type in ["season_planner", "weekly_planner"]:
        timeframe = "12-24 week strategy" if agent_type == "season_planner" else "next 28-day workouts"
        return f"""
## System Role
You are the **{agent_type.replace('_', ' ').title()}**.
- **Input**: `for_{agent_type}` fields from Metrics, Physiology, and Activity experts.
- **Output**: `{agent_type.replace('_planner', '_plan')}` ({timeframe}).
- **Goal**: Translate expert insights into a concrete {timeframe}.
- **Context**: Use the expert signals as your primary constraints and guides."""

    return ""


def get_plotting_instructions(agent_name: str) -> str:
    return f"""
## Visualization Rules
- **Constraint**: Create plots ONLY for unique insights not visible in standard Garmin reports. Max 2 plots.
- **Tool Output**: When the plotting tool succeeds, it returns the exact `plot_id` and reference syntax.
- **Reference**: You MUST reference each created plot EXACTLY ONCE using the exact returned tag `[PLOT:<plot_id>]`.
- **Never Invent IDs**: Do NOT make up semantic or placeholder IDs such as `[PLOT:training_load_overview]` or `[PLOT:{agent_name}_TIMESTAMP_ID]`.
- **Placement**: Place the reference where it best supports your analysis. Do not repeat it."""


def get_available_plot_references_context(available_plot_ids: list[str]) -> str:
    unique_plot_ids = sorted(set(available_plot_ids))

    if not unique_plot_ids:
        return """
## Valid Plot References
- No plots are currently available.
- Do NOT invent any new `[PLOT:...]` references.
"""

    formatted_ids = "\n".join(f"- `{plot_id}`" for plot_id in unique_plot_ids)
    return f"""
## Valid Plot References
- Use ONLY these exact plot IDs if you include plot references.
- Never invent descriptive placeholders such as `[PLOT:training_load_overview]`.
{formatted_ids}
"""


def get_hitl_instructions(agent_name: str) -> str:
    return """
## Human Interaction
- **Questions**: If you need clarification, set `output` to a list of Question items.
- **Otherwise**: Set `output` to your node's normal output schema.
- **Criteria**: Only ask if data is ambiguous or user preference is required. Do not ask for obvious info.
- **Process**: If you ask questions, your execution pauses until the user answers."""

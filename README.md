# Garmin WHOOP AI Coach

This repository is a personal fork of `garmin-ai-coach` adapted to combine Garmin Connect training data with WHOOP recovery data, then run local AI-assisted analysis and planning.

The project remains CLI-first. Garmin is still the primary source for activities and long-term training metrics. WHOOP is used to supplement recovery-related signals such as sleep, HRV, resting heart rate, and day strain.

## Scope

- Extract Garmin Connect activities and training metrics
- Optionally overlay WHOOP recovery, sleep, and strain data
- Run AI analysis and planning locally through the CLI
- Generate local HTML, Markdown, and JSON outputs

## Data Sources

- Garmin: activities, workouts, VO2 max, training load, historical metrics
- WHOOP: recovery, sleep, HRV, resting heart rate, day strain

Current WHOOP support is limited to recovery-oriented data. It does not replace Garmin activity ingestion.

## Main Interface

The primary entry point is:

```bash
python cli/garmin_ai_coach_cli.py --config my_config.yaml
```

If you use Pixi:

```bash
pixi run coach-init my_config.yaml
pixi run coach-cli --config my_config.yaml
```

Detailed CLI configuration and usage notes are in `cli/README.md`.

## Configuration

The project expects a YAML or JSON config file for athlete metadata, extraction windows, competition targets, output paths, and optional WHOOP settings.

Environment variables commonly used with this fork:

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
WHOOP_CLIENT_ID=...
WHOOP_CLIENT_SECRET=...
```

WHOOP uses OAuth. On first use, the CLI opens the browser for consent and stores the resulting token locally for reuse.

## Outputs

A typical run writes artifacts such as:

- `analysis.html`
- `planning.html`
- `summary.json`
- intermediate Markdown files for activity, metrics, physiology, and planning

## Repository Structure

- `cli/`: CLI entrypoint, config template, and usage docs
- `services/garmin/`: Garmin extraction and normalization
- `services/whoop/`: WHOOP auth, extraction, and merge logic
- `services/ai/`: model selection, workflow orchestration, and report generation
- `tests/`: targeted automated tests

## Model Providers

This fork keeps support for both OpenAI and Anthropic. Provider selection depends on the configured AI mode and model mapping in the codebase.

If you want lower-cost runs, disable plotting, disable HITL, and skip synthesis when you do not need the full analysis output.

## Attribution

This repository is derived from the upstream project:

- `leonzzz435/garmin-ai-coach`
- Source: https://github.com/leonzzz435/garmin-ai-coach

This fork keeps the original foundation and adapts it for a Garmin + WHOOP workflow and personal use.

Additional attribution details are in `NOTICE.md`.

## License

This fork is released under the MIT License. See `LICENSE`.

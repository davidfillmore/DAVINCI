# Plume Sentinel

Run a DAVINCI-MONET analysis pipeline in a new Terminal window.

User request: $ARGUMENTS

This command automates the execution of DAVINCI-MONET pipelines in isolated terminal windows. This is particularly useful for long-running analyses or when using the `--show-plots` flag.

## Workflow

1.  **Search & Identify**: 
    -   **Scope**: Only search within the `analyses/plume-sentinel/configs/` directory.
    -   **File Preference**: Prioritize files ending in `-gemini.yaml` as these contain local absolute paths for the current machine.
    -   **Description Mode**: If the user provides keywords (e.g., "modis aod"), search for matching `-gemini.yaml` files.
    -   **Date Mode**: If the user provides a date (e.g., "/plume-sentinel September 9, 2020"), find all `-gemini.yaml` files where the date overlaps with the config's `start_time` and `end_time`.
2.  **Handle Date Overrides**: Create a temporary copy and update the `start_time` and `end_time` fields under the `analysis` section to the requested date.
3.  **Validate Input**: Ensure the final YAML configuration files exist and are valid.
4.  **Verify Environment**: Confirm the `davinci-monet` conda environment is available.
5.  **Prepare Completion Detection**: Remove any stale sentinel files before launching:
    ```bash
    rm -f /tmp/plume-sentinel-done /tmp/plume-sentinel-manifest.txt /tmp/plume-sentinel-window-id
    ```
6.  **Execute in New Window (Serial)**: Use `osascript` to spawn a *single* new terminal and run each identified analysis sequentially. Append the `--show-plots` flag to each command. At the end of the command chain, write a manifest of output files and touch a sentinel file to signal completion (see Platform-Specific Commands below).
7.  **Wait for Completion**: Immediately after launching the terminal, start a background watcher using Bash with `run_in_background: true`:
    ```bash
    while [ ! -f /tmp/plume-sentinel-done ]; do sleep 5; done && echo "Pipelines complete"
    ```
    Continue informing the user that analyses are running. When the background watcher completes, you will be notified automatically — do NOT poll or sleep.
8.  **Close Pipeline Terminal**: Once completion is detected, close the spawned Terminal window:
    ```bash
    osascript -e 'tell application "Terminal" to close (every window whose id is '"$(cat /tmp/plume-sentinel-window-id)"')'
    ```
9.  **Monitor & Summarize** (triggered when watcher completes):
    -   Read `/tmp/plume-sentinel-manifest.txt` to get the list of output image files.
    -   Use the Read tool to inspect each output image.
    -   **Image Analysis Thought Process**: Before filling in the bulletin, share a concise, bulleted "Image Analysis Thought Process". This should explain how you are interpreting the visual data (e.g., identifying plume boundaries, interpreting AOD color scales, or correlating HMS contours with satellite imagery).
    -   **Fill in the bulletin template**: Read the template at `analyses/plume-sentinel/templates/bulletin.template`. Populate all `{{PLACEHOLDER}}` tokens and write the result to `report.txt` (overwrite). The placeholders are:

        **Auto-derived from config/data** (do not hallucinate — extract from the YAML configs):
        | Placeholder | Source |
        |---|---|
        | `{{BULLETIN_ID}}` | Format: `PS-{julian_day}-{REGION_SLUG}-001` (e.g., `PS-2020253-WESTCOAST-001`) |
        | `{{ISSUED_DATE}}` | Today's date |
        | `{{EVENT_DATE}}` | From config `analysis.start_time`, formatted as e.g., `September 9, 2020` |
        | `{{OBSERVATION_TIME}}` | From input filenames/metadata (e.g., `~19:05-19:10 UTC`) |
        | `{{SENSOR_SOURCES}}` | Derived from config input types (e.g., `MODIS Terra AOD, GOES-16 ABI, NOAA HMS`) |

        **AI-analyzed from images** (write in the authoritative tone of "PlumeSentinel AI"):
        | Placeholder | Guidance |
        |---|---|
        | `{{REGION}}` | Geographic region derived from map extent and fire context |
        | `{{SEVERITY}}` | One of: LOW, MODERATE, HIGH, EXTREME — based on holistic assessment of AOD values, HMS coverage, and population impact |
        | `{{SYNOPTIC_OVERVIEW}}` | 2-3 paragraphs: weather pattern, wind regime, fire complexes involved, smoke transport |
        | `{{AOD_ANALYSIS}}` | Structured findings: peak AOD, inland extent, offshore transport, retrieval failures, notable sub-regions |
        | `{{HMS_ANALYSIS}}` | Breakdown by density: Heavy (red), Medium (orange), Light (yellow) contours with spatial extent |
        | `{{HEALTH_IMPACTS}}` | Risk level, estimated AQI, affected population, recommended actions, visibility impacts |
        | `{{ASSESSMENT}}` | Confidence level, classification, historical context |

    -   **Display the completed bulletin** in its entirety at the end of the response.
    -   **MQTT Publishing**: Use an available MQTT client to publish the full text of `report.txt` to the public HiveMQ broker (`broker.hivemq.com`). For example, if using the npm `mqtt` CLI, pipe the file content using the `-s` (stdin) flag: `cat report.txt | npx --yes mqtt pub -t 'plume-sentinel-ai/alerts/test' -h broker.hivemq.com -s`. Publish under the base topic `plume-sentinel-ai`.
        -   *Suggested subtopics*: `plume-sentinel-ai/alerts/test` (for test broadcasts), `plume-sentinel-ai/reports/west-coast` (for region-specific reports), or `plume-sentinel-ai/events/wildfire-smoke` (for event-specific tracking).

### Platform-Specific Commands

#### macOS (Darwin)

Use `osascript` to launch a new Terminal window. The command chain must:
1. `cd` to the project directory
2. Activate the conda environment
3. Run each config sequentially with `--show-plots`
4. Write output file manifest to `/tmp/plume-sentinel-manifest.txt`
5. Touch `/tmp/plume-sentinel-done` to signal completion
6. Return focus to the original window

The osascript captures the spawned window ID by comparing window lists before and after spawning, then writes it to `/tmp/plume-sentinel-window-id` so it can be closed after completion.

**IMPORTANT**: Use absolute paths for config files and output directories. The spawned terminal starts in the user's home directory, and `cd` inside the command chain can silently fail. Absolute paths are reliable.

```bash
osascript -e 'tell application "Terminal"
set orig_window to id of front window
set winsBefore to id of every window
do script "conda activate davinci-monet && osascript -e \"tell application \\\"Terminal\\\" to activate\" && davinci-monet run <absolute_config_path_1> --show-plots && sleep 5 && osascript -e \"tell application \\\"Terminal\\\" to activate\" && davinci-monet run <absolute_config_path_2> --show-plots && ls <absolute_output_dir>/*.png > /tmp/plume-sentinel-manifest.txt && touch /tmp/plume-sentinel-done && sleep 5 && osascript -e \"tell application \\\"Terminal\\\" to set index of window id " & orig_window & " to 1\" -e \"tell application \\\"Terminal\\\" to activate\""
delay 1
set winsAfter to id of every window
repeat with w in winsAfter
if w is not in winsBefore then
do shell script "echo " & w & " > /tmp/plume-sentinel-window-id"
end if
end repeat
end tell'
```

Replace `<absolute_config_path_1>`, `<absolute_config_path_2>`, and `<absolute_output_dir>` with full paths derived from the identified configs (e.g., `/Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/plume-sentinel/configs/modis-aod-truecolor-gemini.yaml`).

### Example Usage

**User:** "/plume-sentinel MODIS AOD September 9, 2020"
**Action:** Identifies `analyses/plume-sentinel/configs/modis-aod-truecolor-gemini.yaml` and launches it.

**User:** "/plume-sentinel September 9, 2020"
**Action:** Finds all `-gemini.yaml` configs in `analyses/plume-sentinel/` covering that date and launches them sequentially in a single new window.

## Guidelines

-   **Priority**: Always use `-gemini.yaml` versions over `.example.yaml` versions when available.
-   **Include environment activation**: Ensure `conda activate davinci-monet` is part of the command.
-   **Real-Time Reporting**: You MUST report all actions you are taking in real-time to the user (e.g., "Searching for configs...", "Validating environment...", "Launching analysis...", "Analyzing results...", "Publishing report...").
-   **Inform the user**: Confirm which specific local analyses have been triggered.

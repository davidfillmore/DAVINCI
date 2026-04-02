---
name: plume-sentinel
description: Runs a DAVINCI-MONET analysis pipeline in a new Terminal window. Use when the user requests an analysis run (e.g., "/plume-sentinel MODIS AOD") and provides a natural language description. The skill will search for matching configuration files in analyses/*/configs/ and handle date-based filtering if provided.
---

# Plume Sentinel

This skill automates the execution of DAVINCI-MONET pipelines in isolated terminal windows. This is particularly useful for long-running analyses or when using the `--open-plots` flag.

## Workflow

1.  **Search & Identify**: 
    -   **Scope**: Only search within the `analyses/plume-sentinel/configs/` directory.
    -   **File Preference**: Prioritize files ending in `-gemini.yaml` as these contain local absolute paths for the current machine.
    -   **Description Mode**: If the user provides keywords (e.g., "modis aod"), search for matching `-gemini.yaml` files.
    -   **Date Mode**: If the user provides a date (e.g., "/plume-sentinel September 9, 2020"), find all `-gemini.yaml` files where the date overlaps with the config's `start_time` and `end_time`.
2.  **Handle Date Overrides**: Create a temporary copy and update the `start_time` and `end_time` fields under the `analysis` section to the requested date.
3.  **Validate Input**: Ensure the final YAML configuration files exist and are valid.
4.  **Verify Environment**: Confirm the `davinci-monet` conda environment is available.
5.  **Execute in New Window (Serial)**: Use `osascript` to spawn a *single* new terminal and run each identified analysis sequentially (one after the other). Append the `--open-plots` flag to each command.
6.  **Monitor & Summarize**:
    -   After the analysis has had time to run, check the `output/` directory for new image files.
    -   **Copy images** to the project temporary directory (e.g., `/Users/fillmore/.gemini/tmp/davinci-monet/plume_sentinel/`) before using `read_file`.
    -   **Session Re-activation**: Ensure the `osascript` command captures the original session window ID and robustly brings it back to the foreground once all analyses complete.
    -   Use `read_file` to inspect the images. Craft the final summary as a **formal meteorological report**, written in the authoritative tone of "PlumeSentinel AI" issuing a high wildfire smoke event alert. The report MUST clearly indicate that it is a **TEST BULLETIN**. Include structured sections for Synoptic Overview, Aerosol Optical Depth (AOD) Analysis, and Hazard Mapping System (HMS) observations based on the visual findings.
    -   **MQTT Publishing**: After crafting the report, use an available MQTT client (e.g., `mosquitto_pub` or a Python script via `paho-mqtt`) to publish the full text of the bulletin to the public HiveMQ broker (`broker.hivemq.com`). Publish it under the base topic `plume-sentinel-ai`. 
        -   *Suggested subtopics*: `plume-sentinel-ai/alerts/test` (for test broadcasts), `plume-sentinel-ai/reports/west-coast` (for region-specific reports), or `plume-sentinel-ai/events/wildfire-smoke` (for event-specific tracking).

### Platform-Specific Commands

#### macOS (Darwin)

Use `osascript` to launch a new Terminal window. To run multiple configs in serial in the same window, and return to the original window after a 5 second delay:

```bash
osascript -e 'tell application "Terminal"
set orig_window to id of front window
do script "cd /Users/fillmore/EarthSystem/DAVINCI-MONET && conda activate davinci-monet && osascript -e \"tell application \\\"Terminal\\\" to activate\" && davinci-monet run <config_path_1> --open-plots && sleep 5 && osascript -e \"tell application \\\"Terminal\\\" to activate\" && davinci-monet run <config_path_2> --open-plots && sleep 5 && osascript -e \"tell application \\\"Terminal\\\" to set index of window id " & orig_window & " to 1\" -e \"tell application \\\"Terminal\\\" to activate\""
end tell'
```

### Example Usage

**User:** "/plume-sentinel MODIS AOD September 9, 2020"
**Action:** Identifies `analyses/plume-sentinel/configs/modis-aod-truecolor-gemini.yaml` and launches it.

**User:** "/plume-sentinel September 9, 2020"
**Action:** Finds all `-gemini.yaml` configs in `analyses/plume-sentinel/` covering that date and launches them sequentially in a single new window.

## Guidelines

-   **Priority**: Always use `-gemini.yaml` versions over `.example.yaml` versions when available.
-   **Include environment activation**: Ensure `conda activate davinci-monet` is part of the command.
-   **Inform the user**: Confirm which specific local analyses have been triggered.

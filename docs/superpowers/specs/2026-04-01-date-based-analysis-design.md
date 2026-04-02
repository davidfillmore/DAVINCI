# Design Spec: Date-Based Analysis Automation (Plume Sentinel)

**Date:** 2026-04-01
**Topic:** Automating multiple DAVINCI-MONET analyses for a specific date with session re-activation and image inspection.

## 1. Overview
Extend the `plume-sentinel` skill to support a "/plume-sentinel <date>" command that automatically discovers, filters, and executes all relevant analyses for that date. This includes spawning new terminal windows, performing the analysis, and re-activating the current session for image-based reporting.

## 2. Architecture & Discovery
- **Discovery Engine**: Recursively search `analyses/*/configs/*.yaml` for configuration files.
- **Overlap Logic**:
    - Parse `start_time` and `end_time` from each YAML file (typically under `analysis:` or `task:` keys).
    - Compare the user-provided date (e.g., "September 9, 2020") with the config's range.
    - If the date falls within [start_time, end_time], the config is marked for execution.

## 3. Execution & Monitoring
- **Temporary Configuration**: For each matching config, create a temporary YAML file with `start_time` and `end_time` overridden to the user's specific date.
- **New Window Command (macOS)**:
    - Use `osascript` to open a new Terminal window.
    - Command string will include:
        1. `cd` to the project root.
        2. `conda activate davinci-monet`.
        3. `davinci-monet run <temp_config_path>`.
        4. **Session Re-activation**: `osascript -e 'tell application "Terminal" to activate'` (or equivalent for the active terminal emulator) to bring the user's focus back to the original session upon completion.

## 4. Post-Analysis Reporting
- **Image Inspection**: After the process finishes in the background window:
    1. Scan the `output/` directory (defined in the config) for new images (`.png`, `.jpg`).
    2. Copy images to the project temporary directory (`/Users/fillmore/.gemini/tmp/davinci-monet/plume_sentinel/`).
    3. Use `read_file` to inspect the images and provide a natural language summary of the visual findings (e.g., "The MODIS AOD plot shows a significant smoke plume over Northern California...").

## 5. Success Criteria
- User can trigger multiple analyses with a single date.
- Only relevant analyses (based on date overlap) are launched.
- The original terminal window is re-activated once the longest-running analysis completes (or as each one finishes).
- Visual summaries are provided for all successful outputs.

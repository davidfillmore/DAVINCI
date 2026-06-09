"""Progress callback routing for pipeline display and reporting."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def create_progress_callback(
    formatter: Any,
    collector: Any | None,
) -> Callable[[str], None]:
    """Create the stage progress callback used by pipeline runs."""

    def callback(msg: str) -> None:
        if collector:
            collector.log_item(msg)

        stripped = msg.strip()
        if stripped.startswith("Loading source:"):
            match = re.match(r"\s*Loading source: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("source", name, int(idx), int(total))
        elif stripped.startswith("Loading model:"):
            match = re.match(r"\s*Loading model: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("model", name, int(idx), int(total))
        elif stripped.startswith("Loading obs:"):
            match = re.match(r"\s*Loading obs: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("obs", name, int(idx), int(total))
        elif stripped.startswith("parallel_start:"):
            match = re.match(r"\s*parallel_start: (\d+)(?:\s*\|\s*(.+))?", msg)
            if match:
                total = int(match.group(1))
                loading_msg = match.group(2)
                formatter.start_parallel(total, loading_msg)
        elif stripped.startswith("parallel_end"):
            formatter.end_parallel()
        elif stripped.startswith("parallel_started:"):
            match = re.match(r"\s*parallel_started: (\S+)", msg)
            if match:
                formatter.parallel_item_started(match.group(1))
        elif stripped.startswith("parallel_completed:"):
            match = re.match(r"\s*parallel_completed: (\S+)(.*)", msg)
            if match:
                name = match.group(1)
                details = match.group(2).strip(" -") if match.group(2) else ""
                formatter.parallel_item_completed("pair", name, details)
        elif stripped.startswith("Pairing:"):
            match = re.match(r"\s*Pairing: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("pair", name, int(idx), int(total), track=False)
        elif stripped.startswith("Paired:"):
            match = re.match(r"\s*Paired: (\S+) \((\d+)/(\d+)\)(.*)", msg)
            if match:
                name, idx, total, details = match.groups()
                details = details.strip(" -") if details else ""
                formatter.item_complete("pair", name, int(idx), int(total), details)
        elif stripped.startswith("Stats:"):
            match = re.match(r"\s*Stats: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("stats", name, int(idx), int(total))
        elif stripped.startswith("Plot:"):
            match = re.match(r"\s*Plot: (\S+) \((\d+)/(\d+)\)", msg)
            if match:
                name, idx, total = match.groups()
                formatter.item_start("plot", name, int(idx), int(total))
        elif stripped.startswith("step:"):
            formatter.step(stripped[5:].strip())
        elif stripped.startswith("done:"):
            formatter.item_done(stripped[5:].strip())

    return callback

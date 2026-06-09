"""Tests for pipeline progress callback routing."""

from __future__ import annotations


class _Formatter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def item_start(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("item_start", args + tuple(sorted(kwargs.items()))))

    def start_parallel(self, *args: object) -> None:
        self.calls.append(("start_parallel", args))

    def end_parallel(self) -> None:
        self.calls.append(("end_parallel", ()))

    def parallel_item_started(self, *args: object) -> None:
        self.calls.append(("parallel_item_started", args))

    def parallel_item_completed(self, *args: object) -> None:
        self.calls.append(("parallel_item_completed", args))

    def step(self, *args: object) -> None:
        self.calls.append(("step", args))

    def item_done(self, *args: object) -> None:
        self.calls.append(("item_done", args))


class _Collector:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log_item(self, message: str) -> None:
        self.messages.append(message)


def test_create_progress_callback_routes_progress_messages() -> None:
    from davinci_monet.pipeline.progress import create_progress_callback

    formatter = _Formatter()
    collector = _Collector()
    callback = create_progress_callback(formatter, collector)

    callback("    Loading source: cam (1/2)")
    callback("    parallel_start: 3 | pairing sources")
    callback("    parallel_started: cam_airnow")
    callback("    parallel_completed: cam_airnow - 1 vars, 10 points")
    callback("    parallel_end")
    callback("step: Computing metrics...")
    callback("done: 1 vars, 6 metrics")

    assert collector.messages[0] == "    Loading source: cam (1/2)"
    assert ("item_start", ("source", "cam", 1, 2)) in formatter.calls
    assert ("start_parallel", (3, "pairing sources")) in formatter.calls
    assert ("parallel_item_started", ("cam_airnow",)) in formatter.calls
    assert (
        "parallel_item_completed",
        ("pair", "cam_airnow", "1 vars, 10 points"),
    ) in formatter.calls
    assert ("end_parallel", ()) in formatter.calls
    assert ("step", ("Computing metrics...",)) in formatter.calls
    assert ("item_done", ("1 vars, 6 metrics",)) in formatter.calls

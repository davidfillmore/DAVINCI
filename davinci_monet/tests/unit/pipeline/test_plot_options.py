"""Tests for plotting-stage option assembly helpers."""

from __future__ import annotations


def test_comparison_plot_options_are_assembled_outside_stage() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_comparison_plot_options

    plot_spec = {
        "show_density": True,
        "marker_size": 12,
        "ignored": "not forwarded",
    }
    analysis_config = {"city_labels": [{"name": "Boulder", "lat": 40.0, "lon": -105.2}]}

    options = build_comparison_plot_options(
        "spatial_overlay",
        plot_spec,
        analysis_config,
        nlevels=21,
    )

    assert options["show_density"] is True
    assert options["marker_size"] == 12
    assert options["city_labels"] == analysis_config["city_labels"]
    assert options["n_levels"] == 21
    assert "ignored" not in options


def test_plot_subtitle_uses_date_range_or_snapshot() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_plot_subtitle

    assert (
        build_plot_subtitle(
            {"start_time": "2024-01-01", "end_time": "2024-01-03"},
            snapshot_timestamp=None,
        )
        == "2024-01-01 - 2024-01-03"
    )
    assert (
        build_plot_subtitle(
            {"start_time": "2024-01-01", "end_time": "2024-01-03"},
            snapshot_timestamp="2024-01-02 12:00 UTC",
        )
        == "2024-01-02 12:00 UTC"
    )

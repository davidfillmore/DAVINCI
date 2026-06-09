"""Output formatters for statistics results.

This module provides functionality for formatting and exporting
statistics results to various formats (CSV, table images, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OutputConfig:
    """Configuration for statistics output.

    Attributes
    ----------
    output_dir : Path | None
        Directory for output files.
    csv_enabled : bool
        If True, write CSV files.
    table_image_enabled : bool
        If True, create table images.
    round_precision : int
        Decimal places for display.
    stat_fullname : bool
        If True, include full names in output.
    stat_fullname_space : bool
        If True, use spaces in full names.
    """

    output_dir: Path | None = None
    csv_enabled: bool = True
    table_image_enabled: bool = False
    round_precision: int = 3
    stat_fullname: bool = True
    stat_fullname_space: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> OutputConfig:
        """Create config from dictionary."""
        config = config_dict.copy()
        if "output_dir" in config and config["output_dir"] is not None:
            config["output_dir"] = Path(config["output_dir"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# Metric Metadata
# =============================================================================

#: Mapping of metric abbreviations to full names
METRIC_FULL_NAMES = {
    "N": "Sample Size",
    "MO": "Mean Reference",
    "MP": "Mean Comparand",
    "STDO": "Reference Standard Deviation",
    "STDP": "Comparand Standard Deviation",
    "MdnO": "Median Reference",
    "MdnP": "Median Comparand",
    "MB": "Mean Bias",
    "MdnB": "Median Bias",
    "NMB": "Normalized Mean Bias",
    "NMdnB": "Normalized Median Bias",
    "FB": "Fractional Bias",
    "MNB": "Mean Normalized Bias",
    "ME": "Mean Error",
    "MdnE": "Median Error",
    "RMSE": "Root Mean Square Error",
    "NME": "Normalized Mean Error",
    "FE": "Fractional Error",
    "MNE": "Mean Normalized Error",
    "R": "Correlation Coefficient",
    "R2": "Coefficient of Determination",
    "IOA": "Index of Agreement",
    "d1": "Modified Index of Agreement",
    "E1": "Modified Coefficient of Efficiency",
    "AC": "Anomaly Correlation",
    "RM": "Mean Ratio",
    "RMdn": "Median Ratio",
}


def get_metric_fullname(metric: str, use_spaces: bool = True) -> str:
    """Get the full name for a metric.

    Parameters
    ----------
    metric
        Metric abbreviation.
    use_spaces
        If True, use spaces. If False, use underscores.

    Returns
    -------
    str
        Full metric name.
    """
    fullname = METRIC_FULL_NAMES.get(metric, metric)
    if not use_spaces:
        fullname = fullname.replace(" ", "_")
    return fullname


# =============================================================================
# Statistics Formatter
# =============================================================================


class StatisticsFormatter:
    """Formatter for statistics output.

    Provides methods for converting statistics DataFrames to
    various output formats.

    Parameters
    ----------
    config
        Output configuration.
    """

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()

    def format_dataframe(
        self,
        stats_df: pd.DataFrame,
        include_fullnames: bool | None = None,
        use_spaces: bool | None = None,
        transpose: bool = False,
    ) -> pd.DataFrame:
        """Format a statistics DataFrame for output.

        Parameters
        ----------
        stats_df
            Statistics DataFrame.
        include_fullnames
            If True, add full name column. Uses config if None.
        use_spaces
            If True, use spaces in full names. Uses config if None.
        transpose
            If True, transpose so metrics are rows.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame.
        """
        include_fullnames = (
            include_fullnames if include_fullnames is not None else self.config.stat_fullname
        )
        use_spaces = use_spaces if use_spaces is not None else self.config.stat_fullname_space

        df = stats_df.copy()

        if transpose:
            # Transpose so metrics are rows
            # Keep only numeric columns for transposition
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df = df[numeric_cols].T
            df.index.name = "Stat_ID"
            df.columns = pd.Index([str(col) for col in df.columns])
            df = df.reset_index()

            if include_fullnames:
                df.insert(
                    1,
                    "Stat_FullName",
                    df["Stat_ID"].apply(lambda x: get_metric_fullname(x, use_spaces)),
                )

        return df

    def to_csv(
        self,
        stats_df: pd.DataFrame,
        output_path: str | Path,
        transpose: bool = True,
        include_fullnames: bool | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write statistics to CSV file.

        Parameters
        ----------
        stats_df
            Statistics DataFrame.
        output_path
            Output file path.
        transpose
            If True, transpose so metrics are rows.
        include_fullnames
            If True, add full name column.
        **kwargs
            Additional arguments for to_csv.

        Returns
        -------
        Path
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.format_dataframe(
            stats_df, include_fullnames=include_fullnames, transpose=transpose
        )

        csv_kwargs = {"index": False, **kwargs}
        df.to_csv(output_path, **csv_kwargs)

        return output_path

    def to_table_image(
        self,
        stats_df: pd.DataFrame,
        output_path: str | Path,
        transpose: bool = True,
        include_fullnames: bool | None = None,
        figsize: tuple[float, float] | None = None,
        fontsize: float = 10,
        dpi: int = 150,
        **kwargs: Any,
    ) -> Path:
        """Create a table image from statistics.

        Parameters
        ----------
        stats_df
            Statistics DataFrame.
        output_path
            Output file path.
        transpose
            If True, transpose so metrics are rows.
        include_fullnames
            If True, add full name column.
        figsize
            Figure size. If None, auto-calculated.
        fontsize
            Font size for table text.
        dpi
            Resolution for output image.
        **kwargs
            Additional arguments.

        Returns
        -------
        Path
            Path to written file.
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.format_dataframe(
            stats_df, include_fullnames=include_fullnames, transpose=transpose
        )

        # Calculate figure size if not provided
        if figsize is None:
            n_rows, n_cols = df.shape
            figsize = (max(4, n_cols * 1.5), max(2, n_rows * 0.4))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis("off")

        # Create table
        table = ax.table(
            cellText=df.to_numpy(dtype=object, copy=False),
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", weight="bold")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

        return output_path

    def to_dict(
        self,
        stats_df: pd.DataFrame,
        orient: Literal["dict", "list", "records", "index"] = "records",
    ) -> dict | list:
        """Convert statistics to dictionary.

        Parameters
        ----------
        stats_df
            Statistics DataFrame.
        orient
            Dictionary orientation (passed to DataFrame.to_dict).

        Returns
        -------
        dict | list
            Dictionary representation.
        """
        return stats_df.to_dict(orient=orient)

    def to_json(
        self,
        stats_df: pd.DataFrame,
        output_path: str | Path,
        orient: str = "records",
        indent: int = 2,
        **kwargs: Any,
    ) -> Path:
        """Write statistics to JSON file.

        Parameters
        ----------
        stats_df
            Statistics DataFrame.
        output_path
            Output file path.
        orient
            JSON orientation.
        indent
            Indentation level.
        **kwargs
            Additional arguments for to_json.

        Returns
        -------
        Path
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats_df.to_json(output_path, orient=orient, indent=indent, **kwargs)

        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================


def write_statistics_csv(
    stats_df: pd.DataFrame,
    output_path: str | Path,
    transpose: bool = True,
    include_fullnames: bool = True,
    **kwargs: Any,
) -> Path:
    """Write statistics DataFrame to CSV.

    Parameters
    ----------
    stats_df
        Statistics DataFrame.
    output_path
        Output file path.
    transpose
        If True, transpose so metrics are rows.
    include_fullnames
        If True, add full name column.
    **kwargs
        Additional CSV options.

    Returns
    -------
    Path
        Path to written file.
    """
    formatter = StatisticsFormatter()
    return formatter.to_csv(
        stats_df,
        output_path,
        transpose=transpose,
        include_fullnames=include_fullnames,
        **kwargs,
    )


def write_statistics_table(
    stats_df: pd.DataFrame,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write statistics DataFrame to table image.

    Parameters
    ----------
    stats_df
        Statistics DataFrame.
    output_path
        Output file path.
    **kwargs
        Additional options.

    Returns
    -------
    Path
        Path to written file.
    """
    formatter = StatisticsFormatter()
    return formatter.to_table_image(stats_df, output_path, **kwargs)


def format_stats_summary(
    stats_dict: Mapping[str, float | None],
    precision: int = 3,
) -> str:
    """Format a statistics dictionary as a summary string.

    Parameters
    ----------
    stats_dict
        Dictionary of metric values.
    precision
        Decimal places for formatting.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    for metric, value in stats_dict.items():
        fullname = get_metric_fullname(metric)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            lines.append(f"{fullname} ({metric}): N/A")
        else:
            lines.append(f"{fullname} ({metric}): {value:.{precision}f}")
    return "\n".join(lines)


def create_comparison_table(
    stats_dfs: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Create a comparison table from multiple statistics DataFrames.

    Parameters
    ----------
    stats_dfs
        Dictionary mapping model/run names to statistics DataFrames.
    metrics
        List of metrics to include. If None, uses all common metrics.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics as rows and models as columns.
    """
    if not stats_dfs:
        return pd.DataFrame()

    # Get common metrics
    all_metrics = set()
    for df in stats_dfs.values():
        all_metrics.update(df.columns)

    if metrics is None:
        metrics = list(all_metrics)

    # Build comparison table
    result = {}
    for name, df in stats_dfs.items():
        if len(df) == 1:
            result[name] = df.iloc[0]
        else:
            # Multiple rows - take first or aggregate
            result[name] = df.iloc[0]

    comparison = pd.DataFrame(result)
    comparison.index.name = "Stat_ID"

    # Add full names
    comparison.insert(0, "Stat_FullName", comparison.index.map(lambda x: get_metric_fullname(x)))
    comparison = comparison.reset_index()

    # Filter to requested metrics
    if metrics:
        requested_metrics = set(metrics)
        comparison = comparison[
            comparison["Stat_ID"].map(lambda stat_id: stat_id in requested_metrics)
        ]

    return comparison

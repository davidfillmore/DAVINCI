"""Helpers for plot titles and date subtitles."""

from __future__ import annotations

import re
from typing import Any

TRAILING_DATE_TITLE_RE = re.compile(
    r"\s+(?:-|--|\u2013|\u2014)\s+"
    r"(?:"
    r"\d{4}-\d{2}-\d{2}(?:\s+(?:-|--|\u2013|\u2014|to)\s+\d{4}-\d{2}-\d{2})?"
    r"|(?:\d{1,2}\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?,?\s+\d{4}"
    r")\s*$",
    re.IGNORECASE,
)

DATE_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:\d{1,2}\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?,?\s+\d{4}"
    r")\s*$",
    re.IGNORECASE,
)


def strip_trailing_date_title(title: str) -> str:
    """Remove a trailing date suffix from a plot title."""
    return TRAILING_DATE_TITLE_RE.sub("", title).rstrip()


def is_date_label(value: Any) -> bool:
    """Return whether a label is a display date."""
    return bool(DATE_LABEL_RE.match(str(value)))


def title_for_labeled_subset(
    base_title: str | None,
    label: Any,
    *,
    label_prefix: str = "",
    separator: str = " - ",
) -> tuple[str, str | None]:
    """Return ``(title, subtitle)`` for a labeled subset such as a flight."""
    label_text = str(label)
    prefix = f"{label_prefix} " if label_prefix else ""
    fallback_title = label_prefix or "Subset"
    if is_date_label(label_text):
        return base_title or fallback_title, label_text
    label_title = f"{prefix}{label_text}"
    return (f"{base_title}{separator}{label_title}" if base_title else label_title), None

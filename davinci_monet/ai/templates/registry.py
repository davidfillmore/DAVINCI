"""Loading and resolution of AI summary comparison templates.

Built-in templates are YAML files under ``data/``; resolution picks a template
for a comparand variable by matching the templates' ``matches`` patterns, with
``generic_eval`` as the fallback. Mirrors the satellite-catalog registry.
"""

from __future__ import annotations

import difflib
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import yaml

from davinci_monet.ai.templates.schema import SummaryTemplate

_DATA_DIR = Path(__file__).parent / "data"
FALLBACK_TEMPLATE = "generic_eval"


class UnknownTemplateError(LookupError):
    """Raised when a template name is not in the registry."""


class TemplateRegistry:
    """An immutable set of templates indexed by name, with variable resolution."""

    def __init__(self, templates: list[SummaryTemplate]) -> None:
        self._by_name: dict[str, SummaryTemplate] = {t.name: t for t in templates}

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def get(self, name: str) -> SummaryTemplate:
        if name in self._by_name:
            return self._by_name[name]
        close = difflib.get_close_matches(name, list(self._by_name), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise UnknownTemplateError(f"Unknown summary template '{name}'.{hint}")

    def merged_with(self, inline: dict[str, dict] | None) -> "TemplateRegistry":
        """Return a new registry with inline templates merged over the built-ins.

        Inline entries are keyed by name; the key is injected as ``name`` when
        the body omits it. An inline name equal to a built-in replaces it.
        """
        templates = dict(self._by_name)
        for name, spec in (inline or {}).items():
            body = dict(spec)
            body.setdefault("name", name)
            templates[name] = SummaryTemplate(**body)
        return TemplateRegistry(list(templates.values()))

    def resolve_for(self, variable: str, *, override: str | None = None) -> SummaryTemplate:
        """Resolve a template: explicit override, else variable match, else fallback."""
        if override:
            return self.get(override)
        var = (variable or "").lower()
        best: tuple[tuple[int, str], SummaryTemplate] | None = None
        for template in self._by_name.values():
            for pattern in template.matches:
                if fnmatchcase(var, pattern.lower()):
                    score = len(pattern.replace("*", "").replace("?", ""))
                    key = (score, template.name)
                    if best is None or key > best[0]:
                        best = (key, template)
        if best is not None:
            return best[1]
        return self.get(FALLBACK_TEMPLATE)


@lru_cache(maxsize=1)
def get_template_registry() -> TemplateRegistry:
    """Load and cache the built-in template library from ``data/*.yaml``."""
    templates: list[SummaryTemplate] = []
    for path in sorted(_DATA_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        templates.append(SummaryTemplate(**raw))
    return TemplateRegistry(templates)


def resolve_template_for(
    variable: str,
    *,
    override: str | None = None,
    inline: dict[str, dict] | None = None,
) -> SummaryTemplate:
    """Resolve a template for ``variable`` against the built-ins plus ``inline``."""
    registry = get_template_registry()
    if inline:
        registry = registry.merged_with(inline)
    return registry.resolve_for(variable, override=override)

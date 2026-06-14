from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path

import yaml

from davinci_monet.datasets.satellite.catalog.schema import ProductEntry

_DATA_DIR = Path(__file__).parent / "data"


class UnknownProductError(LookupError):
    """Raised when a product id/alias is not in the catalog."""


class Catalog:
    def __init__(self, products: list[ProductEntry]) -> None:
        self._by_key: dict[str, ProductEntry] = {}
        for p in products:
            self._by_key[p.product_id] = p
            for alias in p.aliases:
                self._by_key[alias] = p

    def resolve(self, product: str) -> ProductEntry:
        if product in self._by_key:
            return self._by_key[product]
        close = difflib.get_close_matches(product, list(self._by_key), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise UnknownProductError(f"Unknown MODIS/VIIRS product '{product}'.{hint}")

    def product_ids(self) -> list[str]:
        return sorted({p.product_id for p in self._by_key.values()})


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    products: list[ProductEntry] = []
    for yaml_path in sorted(_DATA_DIR.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        for entry in raw.get("products", []):
            products.append(ProductEntry(**entry))
    return Catalog(products)

"""Structures needed for evolutionpy."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .._typing import GENETYPE


class BaseConfig:
    smart_union = True
    arbitrary_types_allowed = True


@dataclass(config=BaseConfig)
class Gene:

    low: GENETYPE
    high: GENETYPE
    value: Optional[GENETYPE] = None
    static: bool = False

    def __post_init__(self):
        if self.low == self.high:
            self.static = True

    def __bool__(self) -> bool:
        return bool(self.low) and bool(self.high)

    @classmethod
    def from_dict(cls, value: Dict[str, GENETYPE], **kwargs) -> Gene:
        """Gene construction from dict."""
        return cls(**value, **kwargs)

    @classmethod
    def from_list(cls, value: List[GENETYPE], **kwargs) -> Gene:
        """Gene construction from dict."""
        low, high = value
        gene_ = {
            "low": low,
            "high": high,
        }
        return cls(**gene_, **kwargs)

    @property
    def random_value(self) -> np.ndarray:
        """Get a uniform random value."""
        return np.random.uniform(low=self.low, high=self.high, size=1)[0]


@dataclass
class Individual:
    genes: List[Gene] = Field(default_factory=list)
    num_of_genes: int = 0

    def __post_init__(self):
        self.num_of_genes = len(self.genes)

    def to_numpy(self) -> np.ndarray:
        individuals_ = [gene.random_value for gene in self.genes]
        return np.asarray(individuals_)

    def __len__(self):
        return self.num_of_genes

    def __bool__(self) -> bool:
        return bool(self.genes)

"""Population Structure."""
from __future__ import annotations

from typing import List

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .gene_space import GeneSpace
from .structures import BaseConfig, Individual


@dataclass(config=BaseConfig)
class Population:
    individuals: List[Individual] = Field(default_factory=list)
    pop_size: int = 0

    def __post_init__(self):
        self.pop_size = len(self.individuals)

    def to_numpy(self):
        populations_ = [indi.to_numpy() for indi in self.individuals]
        return np.asarray(populations_)

    @classmethod
    def from_genespace(
        cls, gene_space: GeneSpace, pop_size: int, **kwargs
    ) -> Population:
        """Initialize population from GeneSpace."""
        individuals = [Individual(genes=gene_space.genes)] * pop_size
        return cls(individuals=individuals)

    def __len__(self):
        return self.pop_size

    def __bool__(self) -> bool:
        return bool(self.individuals)

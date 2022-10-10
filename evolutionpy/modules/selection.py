"""Selection module."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseSelection(ABC):
    """
    Base Class for Selection Module.

    Why selection:
    - Selecting the best individuals in the current generation as parents for\
       producing the offspring of the next generation.
    """

    def __init__(self, population: np.ndarray, num_parents: Optional[int] = None):
        """Initialize BaseSelection."""
        self.population = population
        self.num_parents = num_parents if num_parents else self.population.shape[0] // 2

    @abstractmethod
    def select(
        self, fitness: np.ndarray, num_parents: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract function to implement selection function."""
        raise NotImplementedError


class SteadyStateSelection(BaseSelection):
    """Peform SteadyStateSelection."""

    def select(self, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the parents using the steady-state selection technique.

        Args:
         fitness: The fitness values of the solutions in the current population.
         num_parents: The number of parents to be selected.

        Returns:
         parents: array of the selected parents.
         sorted_fitness: array of sorted fitness for given number of parents.
        """
        fitness_sorted = fitness.argsort()[::-1]
        selected_parents_idx = fitness_sorted[: self.num_parents]
        selected_parents = self.population[selected_parents_idx].copy()
        self.population = selected_parents
        return self.population, selected_parents_idx

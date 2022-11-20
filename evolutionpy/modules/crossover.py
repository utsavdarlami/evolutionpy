from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from .utils import get_nparange


class BaseCrossover(ABC):
    """
    Base Crossover class.

    References:
    - https://www.wikiwand.com/en/Crossover_(genetic_algorithm)
    """

    @abstractmethod
    def _crossover(
        self, parents: np.ndarray, offspring_size: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def crossover(
        self, parents: np.ndarray, offspring_size: Optional[int] = None
    ) -> np.ndarray:
        """Crossover func."""
        # logger.info(f"Performing {self.__class__.__name__} on the parents")
        return self._crossover(parents, offspring_size)

    def __repr__(self):
        """Class representation."""
        return f"{type(self).__name__}()"


class KPointCrossover(BaseCrossover):
    """
    k-point crossover.

    References:
    - https://www.wikiwand.com/en/Crossover_(genetic_algorithm)#/Two-point_and_k-point_crossover
    """

    def __init__(self, k: int, crossover_probability: float = 0.0):
        """
        Initialize class.

        Args
        ____
        k: positive integer
        """
        self.crossover_probability = crossover_probability
        if not isinstance(k, int) or (k <= 0):
            raise ValueError(
                f"k must be positive integer, but {k} of {type(k)} was passed."
            )
        self.k = k
        # if not self.crossover_probability:
        # logger.info("No parent's selected for crossover")

    def get_parents_pair(
        self, parents: np.ndarray, parents_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply func.

        Args:
        -----
        np.ndarray

        Return:
        -------
        np.ndarray
        """
        parents_1 = parents[parents_idx[:, 0]]
        parents_2 = parents[parents_idx[:, 1]]
        paired_1 = np.hstack((parents_1, parents_2))
        paired_2 = np.hstack((parents_2, parents_1))
        return paired_1, paired_2

    def mask_idx(
        self, points: np.ndarray, constant_values: np.ndarray, num_of_genes: int
    ) -> np.ndarray:
        """Get mask index from the given crosspoint range."""
        strides = sliding_window_view(points, 2)[::2]
        indices = get_nparange(strides[0])
        for stride in strides[1:]:
            indices = np.concatenate((indices, get_nparange(stride)))
            if len(indices) < num_of_genes:
                indices = np.setxor1d(indices, constant_values)
        return indices

    def get_crossover_mask(
        self, crossover_point: np.ndarray, num_of_genes: int
    ) -> np.ndarray:
        """
        Get the crossover mask to represent which genes to crossover.

        Args:
        -----
        np.ndararay: crossover point idx to get the crossover-mask

        Return:
        -------
        np.ndarray with dtype as bool
        """
        offspring_size = crossover_point.shape[0]
        end_crossover_point = crossover_point + num_of_genes
        constant_values = np.arange(num_of_genes, num_of_genes * 2)
        # get the range to represent the crossover section
        range_between_crossover_point = np.hstack(
            (crossover_point, end_crossover_point)
        )
        # indexes where we want to keep the original value
        non_crossover_idx = np.apply_along_axis(
            self.mask_idx,
            1,
            range_between_crossover_point,
            constant_values=constant_values,
            num_of_genes=num_of_genes,
        )

        rows_idx = np.arange(0, offspring_size).reshape(offspring_size, 1)
        crossover_mask = np.ones((offspring_size, num_of_genes * 2), dtype=bool)
        crossover_mask[rows_idx, non_crossover_idx] = False
        return crossover_mask

    def _crossover(
        self, parents: np.ndarray, offspring_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply the k-point crossover.

        Selects point(s) randomly at which crossover takes place between the pairs of parents.

        Args:
            - parents: The parents to mate for producing the offspring.
            - offspring_size: The size of the offspring to produce.
        Return:
            - array the produced offspring.
        """
        if not self.crossover_probability:
            return parents

        num_of_parents, num_of_genes = parents.shape
        offsprings = parents.copy()

        if not offspring_size or offspring_size * 2 > num_of_parents:
            offspring_size = num_of_parents // 2

        point_start = 1
        if num_of_genes <= 1:
            point_start = 0

        crossover_point = np.random.randint(
            point_start, num_of_genes, size=(offspring_size, self.k)
        )
        crossover_mask = self.get_crossover_mask(crossover_point, num_of_genes)

        parents_idx_pair = np.vstack(
            (np.arange(0, num_of_parents), np.arange(1, num_of_parents + 1))
        ).T
        parents_idx_for_crossover = parents_idx_pair[:offspring_size]
        paired_1, paired_2 = self.get_parents_pair(parents, parents_idx_for_crossover)

        cross_prob_idx = np.where(
            np.random.rand(paired_1.shape[0]) < self.crossover_probability
        )[0]
        paired_1 = paired_1[cross_prob_idx]
        paired_2 = paired_2[cross_prob_idx]
        crossover_mask = crossover_mask[cross_prob_idx]

        if crossover_mask.size < 0:
            return offsprings

        new_offspring_size = crossover_mask.shape[0]
        offsprings[cross_prob_idx] = paired_1[crossover_mask].reshape(
            new_offspring_size, num_of_genes
        )
        offsprings[cross_prob_idx + 1] = paired_2[crossover_mask].reshape(
            new_offspring_size, num_of_genes
        )
        return offsprings

    def __repr__(self):
        """Class representation."""
        return f"{type(self).__name__}(k={self.k}, crossover_probability={self.crossover_probability})"


class SinglePointCrossover(KPointCrossover):
    """
    SinglePointCrossover.

    i.e k=1 in K-point Crossover.
    """

    k = 1

    def __init__(self, crossover_probability: float = 0.0):
        """
        Initialize class.

        Args
        ____
        """
        super().__init__(k=self.k, crossover_probability=crossover_probability)

    def __repr__(self):
        """Class representation."""
        return (
            f"{type(self).__name__}(crossover_probability={self.crossover_probability})"
        )


class TwoPointCrossover(KPointCrossover):
    """
    TwoPointCrossover.

    i.e k=2 in K-point Crossover.
    """

    k = 2

    def __init__(self, crossover_probability: float = 0.0):
        """
        Initialize class.

        Args
        ____
        """
        super().__init__(k=self.k, crossover_probability=crossover_probability)

    def __repr__(self):
        """Class representation."""
        return (
            f"{type(self).__name__}(crossover_probability={self.crossover_probability})"
        )

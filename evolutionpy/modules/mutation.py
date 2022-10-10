from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evolutionpy.structures import GeneSpace, Population


class BaseMutation(ABC):
    """Base class for mutation."""

    def __init__(self, mutation_rate: float = 0.0, replace: bool = True):
        """
        Initialize class.

        Args:
        -----
        mutation_rate : rate by which mutation occurs
        replace : if True will replace the gene value with new value else does
        averaging of the both values.
        """
        self.mutation_rate = mutation_rate
        self.replace = replace

    def mutation_mask(self, pop_size: int, num_of_genes: int) -> np.ndarray:
        """
        Create a mutation mask.

        It helps to determine which gene for each population should be mutated
        or not based on the `mutation_rate`

        Args:
        -----
        pop_size : population size
        num_of_genes: number of genes

        Return:
        -------
        numpy array with bool values to represent which genes are to be mutate
        True value represents the gene should be mutate else not.
        """
        mutation_mask = np.random.rand(pop_size, num_of_genes) < self.mutation_rate
        return mutation_mask

    def get_mutants(self, pop_size: int, gene_space: GeneSpace) -> np.ndarray:
        """
        Get new population from gene space.

        Args:
        -----
        pop_size :
        gene_space :

        Return:
        -------
        a numpy array having new population
        """
        pass
        if not isinstance(gene_space, GeneSpace):
            raise ValueError(
                f"Expected gene_space to be type of GeneSpace, but got {type(gene_space)}"
            )
        mutating_population = Population.from_genespace(
            pop_size=pop_size, gene_space=gene_space
        ).to_numpy()
        return mutating_population

    @abstractmethod
    def mutate(self, offspring: np.ndarray, gene_space: GeneSpace) -> np.ndarray:
        """Perform mutation."""
        raise NotImplementedError


class RandomMutation(BaseMutation):
    """Perform Random Mutation."""

    def mutate(self, offspring: np.ndarray, gene_space: GeneSpace) -> np.ndarray:
        """
        Apply the random mutation which changes the values of a number of genes randomly.

        Args:
            - offspring: The offspring to mutate.
        Returns:
            - array of the mutated offspring.
        """
        if self.mutation_rate <= 0.0:
            return offspring

        pop_size, num_of_genes = offspring.shape
        mutants = self.get_mutants(pop_size, gene_space)
        mask = self.mutation_mask(pop_size, num_of_genes)
        if self.replace:
            offspring[mask] = mutants[mask]
        else:
            offspring[mask] = (offspring[mask] + mutants[mask]) / 2
        return offspring

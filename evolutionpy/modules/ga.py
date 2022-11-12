"""Genetic Algorithm Custom Flow."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

import numpy as np
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

from evolutionpy.structures import GeneSpace, Population

from ._base import BaseOptimizer
from .crossover import BaseCrossover, SinglePointCrossover
from .mutation import BaseMutation, RandomMutation
from .selection import BaseSelection, SteadyStateSelection


def default_crossover(crossover_probability: float) -> BaseCrossover:
    """Return a default crossover operator."""
    _crossover = SinglePointCrossover(crossover_probability=crossover_probability)
    logger.warning(f"Crossover Operator was not defined, using fallback {_crossover}")
    return _crossover


def default_mutation(mutation_rate: float) -> BaseMutation:
    """Return a default mutation operator."""
    _mutation = RandomMutation(mutation_rate=mutation_rate)
    logger.warning(f"Mutation Operator was not defined, using fallback {_mutation}")
    return _mutation


def default_selection(population: np.ndarray, population_size: int) -> BaseSelection:
    """Return a default selection operator."""
    _selection = SteadyStateSelection(population, population_size)
    logger.warning(f"Selection Operator was not defined, using fallback {_selection}")
    return _selection


class BaseGA(BaseOptimizer):
    """
    BAse GA to Work with for now.

    TODO:
      - Log which operator are being used and there args value
    """

    def __init__(
        self,
        gene_space: Any,
        fitness_func: Callable,
        population_size: int,
        num_generations: int,
        mutation_rate: float = 0.1,
        crossover_probability: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize class.

        Args:
        -----
        None
        """
        self.fitness_func = fitness_func
        self.num_generations = num_generations
        self.population_size = population_size

        if isinstance(gene_space, GeneSpace):
            self.gene_space = gene_space
        else:
            self.gene_space = GeneSpace(gene_space, num_of_genes=num_generations)

        self.population = Population.from_genespace(
            pop_size=population_size, gene_space=self.gene_space
        ).to_numpy()

        self.crossover_probability = crossover_probability
        self.mutation_rate = mutation_rate

        self.best_fitness: Optional[Union[int, float]] = None
        self.best_individual: Optional[np.ndarray] = None
        self.best_generation: Optional[int] = None
        self.fitness_history: List[Union[int, float]] = []
        self.best_individual_history: List[np.ndarray] = []

    def default_operator(self):
        """Set Default operators."""
        self.selection = default_selection(self.population, self.population_size)
        self.crossover = default_crossover(
            crossover_probability=self.crossover_probability
        )
        self.mutation = default_mutation(mutation_rate=self.mutation_rate)

    def set_operators(self):
        """Set Operators."""
        self.default_operator()

    def set_best_solution(self, fitness: np.ndarray, **kwargs):
        """
        Set best solution.

        Args:
        -----
        fitness: array contain fitness for each individual in a population

        Return:
        -------
        None
        """
        # gen_idx = kwargs.get("gen_idx", 1)
        max_idx = fitness.argmax()
        self.fitness_history.append(fitness.max())
        self.best_individual_history.append(self.population[max_idx])
        self.best_fitness = max(self.fitness_history)
        self.best_generation = self.fitness_history.index(self.best_fitness)
        self.best_individual = self.best_individual_history[self.best_generation]

    def pop_fitness(
        self, fitness_func: Callable[[np.ndarray], float], **kwargs
    ) -> np.ndarray:
        """Calculate the population fitness given the fitness function."""
        pop_fitness = np.apply_along_axis(fitness_func, 1, self.population, **kwargs)
        self.set_best_solution(pop_fitness, **kwargs)
        return pop_fitness

    def update(self, gen_idx: int, **kwargs):
        """
        Run the GA for a particular instance of poulation.

        Args:
        -----
        gen_idx: int
        - Represents generation index

        Return:
        -------
        None
        """
        # fitness function
        fitness = self.pop_fitness(
            fitness_func=self.fitness_func, gen_idx=gen_idx, **kwargs
        )

        # selection
        selected_parents, selected_parent_idx = self.selection.select(fitness)
        # crossover
        offsprings = self.crossover.crossover(selected_parents)
        # mutation
        if self.mutation:
            offsprings = self.mutation.mutate(offsprings, gene_space=self.gene_space)

        self.population = offsprings

    def validate_selection(self):
        """Validate the `selection` operator."""
        if not hasattr(self, "selection"):
            self.selection = default_selection(self.population, self.population_size)
        assert isinstance(self.selection, BaseSelection)

    def validate_crossover(self):
        """Validate the `selection` operator."""
        if not hasattr(self, "crossover"):
            self.crossover = default_crossover(self.crossover_probability)
        assert isinstance(self.crossover, BaseCrossover)

    def validate_mutation(self):
        """Validate the `selection` operator."""
        if not hasattr(self, "mutation"):
            self.mutation = default_mutation(self.mutation_rate)
        assert isinstance(self.mutation, BaseMutation)

    def validate_operators(self):
        """Validate the operators used in GA."""
        self.validate_selection()
        self.validate_crossover()
        self.validate_mutation()

    def run(self, **kwargs):
        """
        Run the GA.

        Args:
        -----
        None

        Return:
        -------
        None
        """
        self.init_metrics_container()
        self.set_operators()
        self.validate_operators()

        text_column = TextColumn("{task.description}", table_column=Column(ratio=1))
        bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1))
        task_processed_column = TaskProgressColumn()
        mofn_column = MofNCompleteColumn()
        time_elpased_column = TimeElapsedColumn()
        progress = Progress(
            mofn_column,
            bar_column,
            task_processed_column,
            time_elpased_column,
            text_column,
        )

        with progress:
            task_id = progress.add_task(
                "[green]Processing...", total=self.num_generations
            )
            progress.start_task(task_id)
            for gen_idx in range(self.num_generations):
                self.update(gen_idx, **kwargs)
                task_description = f"Generation : {gen_idx+1} \nBest Fitness : {self.best_fitness} \nIndividual : {self.best_individual} \nBest Generation Index : {self.best_generation}"
                progress.update(
                    task_id, advance=1, description=task_description, refresh=True
                )

        _ = self.pop_fitness(self.fitness_func, **kwargs)
        logger.info("Genetic Algorithm Run Completed")
        logger.info(
            f"Best Fitness is {self.best_fitness} obtained for {self.best_individual} indiviual at {self.best_generation} generation"
        )

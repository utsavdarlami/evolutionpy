"""Example for baseGA."""
import numpy as np

from evolutionpy.modules.crossover import SinglePointCrossover
from evolutionpy.modules.ga import BaseGA
from evolutionpy.modules.mutation import RandomMutation
from evolutionpy.modules.selection import SteadyStateSelection
from evolutionpy.structures import GeneSpace


# Given the following function:
# f(x,y) = xy - x^2 - y^2 - 2x - 2y + 4
# Here, f(x,y) attends maximum value f_max=8 at the point (x,y) = (-2,-2)
def fitness_func(individual: np.ndarray, **kwargs) -> float:
    """Fitness function for above equations."""
    x = individual[0]
    y = individual[1]
    return x * y - x**2 - y**2 - 2 * x - 2 * y + 4


class CustomGA(BaseGA):
    """Custom GA for the above problem."""

    def set_operators(self):
        """Set up operators."""
        # self.default_operator()
        self.selection = SteadyStateSelection(self.population, self.population_size)
        self.crossover_probability = 0.3
        self.crossover = SinglePointCrossover(
            crossover_probability=self.crossover_probability
        )
        self.mutation_rate = 0.1
        self.mutation = RandomMutation(mutation_rate=self.mutation_rate)


if __name__ == "__main__":
    num_generations = 100
    population_size = 100  # Number of solutions in the population.

    bounds = [[-10, 10], [-10, 10]]

    gene_space = GeneSpace(bounds)

    custom_ga = CustomGA(
        gene_space=gene_space,
        fitness_func=fitness_func,
        num_generations=num_generations,
        population_size=population_size,
    )
    # Running the GA to optimize the parameters of the function.
    custom_ga.run()

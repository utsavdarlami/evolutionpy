"""
Segment passed image.

>> python example_segmentation.py <img_path> <num_segments> <num_generation>
"""
import argparse
import os
import numpy as np
from scipy import stats
from skimage import color, io
from loguru import logger

# evolutionpy
from evolutionpy.modules.crossover import SinglePointCrossover
from evolutionpy.modules.ga import BaseGA
from evolutionpy.modules.mutation import RandomMutation
from evolutionpy.modules.selection import SteadyStateSelection
from evolutionpy.structures import GeneSpace


def is_sorted(a: np.ndarray) -> bool:
    """Check whether the array is in sorted or not in ascending order."""
    if not isinstance(a, np.ndarray):
        return False
    return np.all(a[:-1] <= a[1:])


def entropy_of_pixel(pixels: np.ndarray, base=None) -> float:
    """Compute entropy of pixel distribution."""
    _, counts = np.unique(pixels, return_counts=True)
    ent = stats.entropy(counts)
    return ent


def calculate_segment_entropy(image: np.ndarray,
                              thresh: np.ndarray = None) -> float:
    """
    Caculate the average entropy of the regions segmented \
    using the thresholds.

    Args:
      image:
      thresh:
    Results:
       Avg of the entropy for `len(thresh) + 1` segments
    """
    if not isinstance(thresh, np.ndarray):
        return 0.0

    regions = np.digitize(image, bins=thresh)
    sum_of_entropies = 0.0
    for idx in range(len(thresh) + 1):
        segmented_pixels = image[regions == idx]
        region_entropy = entropy_of_pixel(segmented_pixels)
        sum_of_entropies += region_entropy

    return sum_of_entropies/(len(thresh)+1)


def fitness_func(individual: np.ndarray, **kwargs) -> float:
    """
    Calculate the fitness value of a \
    solution/state in the current population.

    Args:
    Returns:
    """
    global x_image
    if not is_sorted(individual):
        return 0.0
    # Entropy of the segmented pixel could be a one measure for the fitness
    # we can also add variance as the mesasure of fitness
    fitness = calculate_segment_entropy(x_image,
                                        thresh=individual)
    return fitness


def segment(x_image: np.ndarray, best_individual: np.ndarray) -> np.ndarray:
    """Perform Segmentation."""
    # Running the GA to optimize the thresholds.
    if not isinstance(x_image, np.ndarray):
        raise TypeError(f"x_image should not be Type of {type(x_image)}")
    segmented_image = np.digitize(x_image, bins=best_individual)
    return segmented_image


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

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Image path")
    parser.add_argument("num_segments", type=int, help="Number of segments", default=2)
    parser.add_argument("num_gen", type=int, help="Number of Generation", default=50)
    args = parser.parse_args()

    image_path = args.path
    n_classes = args.num_segments
    num_generations = args.num_gen
    population_size = 100  # Number of solutions in the population.

    bounds = {'low': 0, 'high': 255}

    gene_space = GeneSpace(bounds, num_of_genes=n_classes-1)

    logger.info(image_path)

    x_image = io.imread(image_path, as_gray=True)

    image_segmenter = CustomGA(fitness_func=fitness_func,
                               gene_space=gene_space,
                               num_generations=num_generations,
                               population_size=population_size,
                               )

    logger.info("Performing Segmentation of image")
    image_segmenter.run()

    seg_img = segment(x_image, image_segmenter.best_individual)

    image_name = os.path.split(image_path)
    output_name = "segmented_" + image_name[1]

    seg_img = color.label2rgb(seg_img, bg_label=0)
    seg_img = (seg_img * 255).clip(0, 255).round().astype(np.uint8)

    logger.info(f"Writing the segmented image to {output_name}")
    io.imsave(output_name, seg_img)
    logger.info("| Done |")

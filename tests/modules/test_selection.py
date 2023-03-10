import numpy as np
import pytest

from evolutionpy.modules.selection import SteadyStateSelection

# from pydantic import ValidationError


selection_test_data = [
    (([1.7, 2.1, -2.1, 1.9, 2.0], 2), ([2.1, 2.0], [1, 4])),
    (([1.8, 2.1, 2.1, -2.1, 1.9, -2.0], 3), ([2.1, 2.1, 1.9], [2, 1, 4])),
]


@pytest.mark.parametrize("value, expected", selection_test_data)
def test_population_based_on_shape(value, expected):
    pop, num_parents = value
    pop_np = np.array(pop)
    sss = SteadyStateSelection(population=pop_np, num_parents=num_parents)
    new_pop, pop_idx = sss.select(fitness=pop_np)
    assert np.array_equal(new_pop, np.array(expected[0]))
    assert np.array_equal(pop_idx, np.array(expected[1]))

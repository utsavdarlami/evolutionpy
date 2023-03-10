import pytest
from pydantic import ValidationError

from evolutionpy.structures import GeneSpace, Population

population_shape_test_data = [
    ((([[0, 2], [4, 6]], None), 5), (5, 2)),
    ((({"low": 1, "high": 2}, 3), 5), (5, 3)),
    ((([["a", 2], [4, 6]], None), 5), ValidationError),
]


@pytest.mark.parametrize("value, expected", population_shape_test_data)
def test_population_based_on_shape(value, expected):
    try:
        gene_space_data, pop_size = value
        space_, num_of_genes = gene_space_data
        gene_space = GeneSpace(space_, num_of_genes=num_of_genes)
        pop_ = Population.from_genespace(pop_size=pop_size, gene_space=gene_space)
        pop_np_arr = pop_.to_numpy()
        assert pop_np_arr.shape == expected
    except Exception as e:
        assert type(e) == expected

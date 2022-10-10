import pytest
from pydantic import ValidationError

from evolutionpy.structures import Gene, GeneSpace

genespace_length_test_data = [
    (([[0, 2], [4, 6]], None), 2),
    (([["a", 2], [4, 6]], None), ValidationError),
    (({"low": 1, "high": 2}, 3), 3),
    (([{"low": 1, "high": 2}, {"low": 2, "high": 8}], None), 2),
]


genespace_from_dict_test_data = [
    (
        ({"low": 1, "high": 2}, 3),
        [
            Gene(low=1, high=2, value=None, static=False),
            Gene(low=1, high=2, value=None, static=False),
            Gene(low=1, high=2, value=None, static=False),
        ],
    ),
]

genespace_from_list_of_dict_test_data = [
    (
        [{"low": 0, "high": 2}, {"low": 4, "high": 6}],
        [
            Gene(low=0, high=2),
            Gene(low=4, high=6),
        ],
    ),
]

genespace_from_list_of_list_test_data = [
    (
        [[0, 2], [4, 6]],
        [
            Gene(low=0, high=2),
            Gene(low=4, high=6),
        ],
    ),
]

genespace_from_list_test_data = [
    *genespace_from_list_of_list_test_data,
    *genespace_from_list_of_dict_test_data,
]


@pytest.mark.parametrize("value, expected", genespace_length_test_data)
def test_genespace_based_on_len(value, expected):
    try:
        data, num_of_genes = value
        gene_space = GeneSpace(data, num_of_genes=num_of_genes)
        assert len(gene_space) == expected
    except Exception as e:
        assert type(e) == expected


@pytest.mark.parametrize("value, expected", genespace_from_dict_test_data)
def test_from_dict_genespace(value, expected):
    try:
        data, num_of_genes = value
        gene_space = GeneSpace(data, num_of_genes=num_of_genes)
        assert len(gene_space) == len(expected)
        for v, e in zip(gene_space, expected):
            assert v == e
    except Exception as e:
        assert type(e) == expected


@pytest.mark.parametrize("value, expected", genespace_from_list_test_data)
def test_from_list_genespace(value, expected):
    try:
        gene_space = GeneSpace(value)
        assert len(gene_space) == len(expected)
        for v, e in zip(gene_space, expected):
            assert v == e
    except Exception as e:
        assert type(e) == expected

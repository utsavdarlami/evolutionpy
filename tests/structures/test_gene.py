import pytest
from pydantic import ValidationError

from evolutionpy.structures import Gene

gene_from_list_test_data = [
    ([1, 2], (1, 2, None, False)),
    ([2.0, 2.0], (2.0, 2.0, None, True)),
    ([1.0, 2], (1.0, 2, None, False)),
    ([1.0, None], ValidationError),
    ([[1.0], 2.0], ValidationError),
    (["a", 2.0], ValidationError),
    ([1.0, 1, 2], ValueError),
]

gene_from_dict_test_data = [
    ({"low": 10, "high": 12}, (10, 12, None)),
    ({"low": 10, "high": 1.0}, (10, 1.0, None)),
    ({"high": "0.0", "low": 10}, ValidationError),
    ({"high": "0.0", "low": 10}, ValidationError),
    ({"value": [10]}, TypeError),
    ({"value": "a"}, TypeError),
]


@pytest.mark.parametrize("value, expected", gene_from_list_test_data)
def test_gene_from_list(value, expected):
    try:
        gene = Gene.from_list(value)
        assert (gene.low, gene.high, gene.value, gene.static) == expected
    except Exception as e:
        assert type(e) == expected


@pytest.mark.parametrize("value, expected", gene_from_dict_test_data)
def test_gene_from_dict(value, expected):
    try:
        gene = Gene(**value)
        assert (gene.low, gene.high, gene.value) == expected
    except Exception as e:
        assert type(e) == expected

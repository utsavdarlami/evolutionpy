"""Class construction function."""
from typing import List

from loguru import logger

from .._typing import DICTGENETYPE, LISTGENETYPE
from ..structures.structures import Gene
from ..structures.types import ANYGENETYPE
from .type_check import is_dict_list, is_gene_list, is_nested_list_like


def list_of_list_to_genes(data: List[LISTGENETYPE], **kwargs) -> List[Gene]:
    """Form list of genes from list of list."""
    genes = [Gene.from_list(value) for value in data]
    return genes


def list_of_dict_to_genes(data: List[DICTGENETYPE]) -> List[Gene]:
    """Form list genes from list of dict."""
    genes = [Gene(**value) for value in data]
    return genes


def list_to_genes(data: List[ANYGENETYPE], **kwargs) -> List[Gene]:
    """Form list genes from data list."""
    _genes: List[Gene] = []
    if is_gene_list(data):
        _genes = data
    elif is_nested_list_like(data):
        _genes = list_of_list_to_genes(data, **kwargs)
    elif is_dict_list(data):
        _genes = list_of_dict_to_genes(data, **kwargs)
    else:
        logger.error(f"Invalid data type for data, {type(data[-1])} was passed.")
        _genes = []
    return _genes


def dict_to_genes(data: DICTGENETYPE, num_of_genes: int = None, **kwargs) -> List[Gene]:
    """Forming gene space from dict."""
    if not num_of_genes or num_of_genes < 1:
        raise ValueError(
            f"Number of genes should be greater than 0 but {num_of_genes} was passed"
        )
    low = data.get("low", None)
    high = data.get("high", None)
    if low is None or high is None:
        raise ValueError(
            f"high and low should not be None but {high} and {low} was passed respectively."
        )
    # skip = value.get("skip", None)
    values: List = []
    for i in range(int(num_of_genes)):
        values.append([low, high])

    genes = list_of_list_to_genes(values, **kwargs)
    return genes

"""
To check the data type.

Credit:
 - https://github.com/pandas-dev/pandas/blob/main/pandas/core/dtypes/inference.py
"""
from numbers import Number
from typing import Any

import numpy as np

from ..structures.structures import Gene


def is_list_like(obj: Any) -> bool:
    """
    Checks if the obj is a list type of not.
    Examples
    --------
    >>> is_list_like([1, 2, 3])
    True
    >>> is_list_like("foo")
    False
    >>> is_list_like(1)
    False
    """
    return isinstance(obj, list)


def is_number(obj: Any) -> bool:
    """
    Check if the object is a number.

    Returns True when the object is a number, and False if is not.
    Parameters
    ----------
    obj : any type
        The object to check if is a number.
    Returns
    -------
    is_number : bool
        Whether `obj` is a number or not.
    Examples
    --------
    >>> is_number(1)
    True
    >>> is_number(7.15)
    True
    Booleans are valid because they are int subclass.
    >>> is_number(False)
    True
    >>> is_number("foo")
    False
    >>> is_number("5")
    False
    """
    return isinstance(obj, (Number, np.number))


def is_dict_like(obj: Any) -> bool:
    """
    Check if the object is dict-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_dict_like : bool
        Whether `obj` has dict-like properties.

    Examples
    --------
    >>> is_dict_like({1: 2})
    True
    >>> is_dict_like([1, 2, 3])
    False
    >>> is_dict_like(dict)
    False
    >>> is_dict_like(dict())
    True
    """
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return all(hasattr(obj, attr) for attr in dict_like_attrs) and not isinstance(
        obj, type
    )


def is_nested_list_like(obj: Any) -> bool:
    """
    Check if the object is list-like, and that all of its elements
    are also list-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_list_like : bool
        Whether `obj` has list-like properties.

    Examples
    --------
    >>> is_nested_list_like([[1, 2, 3]])
    True
    >>> is_nested_list_like([{1, 2, 3}, {1, 2, 3}])
    True
    >>> is_nested_list_like(["foo"])
    False
    >>> is_nested_list_like([])
    False
    >>> is_nested_list_like([[1, 2, 3], 1])
    False

    Notes
    -----
    This won't reliably detect whether a consumable iterator (e. g.
    a generator) is a nested-list-like without consuming the iterator.
    To avoid consuming it, we always return False if the outer container
    doesn't define `__len__`.
    """
    return (
        is_list_like(obj)
        and hasattr(obj, "__len__")
        and len(obj) > 0
        and all(is_list_like(item) for item in obj)
    )


def is_gene_like(obj: Any) -> bool:
    """TO check if the obj is of type Gene."""
    return isinstance(obj, Gene)


def is_gene_list(obj: Any) -> bool:
    """
    Check if the object is list of gene.

    That all of its elements are of type Gene.
    """
    return (
        is_list_like(obj)
        and len(obj) > 0
        and hasattr(obj, "__len__")
        and all(is_gene_like(item) for item in obj)
    )


def is_dict_list(obj: Any) -> bool:
    """
    Check if the object is list of dict.

    That all of its elements are of type dict.
    """
    return (
        is_list_like(obj)
        and len(obj) > 0
        and hasattr(obj, "__len__")
        and all(is_dict_like(item) for item in obj)
    )

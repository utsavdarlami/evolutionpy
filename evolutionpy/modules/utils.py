import numpy as np


def get_nparange(a: np.ndarray) -> np.ndarray:
    """
    Apply func.

    Args:
    -----
    a: np.ndarray

    Return:
    -------
    np.ndarray

    Example
    -------
    >> [3,7] -> [3, 4, 5, 6]
    """
    if a[0] == a[1]:
        return np.arange(a[0], a[1] + 1)
    return np.arange(a[0], a[1])

"""custom typing."""
from __future__ import annotations

from numbers import Number
from typing import Dict, List, Union

import numpy as np

GENETYPE = Union[Number, np.number]
DICTGENETYPE = Dict[str, GENETYPE]
LISTGENETYPE = List[GENETYPE]

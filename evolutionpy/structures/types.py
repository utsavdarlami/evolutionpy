"""custom types for structures."""
from __future__ import annotations

from typing import Union

from .._typing import DICTGENETYPE, LISTGENETYPE
from .structures import Gene

ANYGENETYPE = Union[DICTGENETYPE, LISTGENETYPE, Gene]

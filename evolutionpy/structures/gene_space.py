from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from ..internals.construction import dict_to_genes, list_to_genes
from .structures import Gene


class GeneSpace:
    """
    Space of values of each gene.

    TODO:
     - Error handling
     - Support numpy instances for gene space creation
    """

    def __init__(
        self,
        data: Any = None,
        num_of_genes: int = None,
        copy: bool | None = None,
        **kwargs,
    ):
        """Initialize class."""

        if data is None:
            _genes = []

        if not isinstance(data, (list, dict)):
            raise ValueError(f"Expected of type dict or list but found {type(data)}")

        if isinstance(data, list):
            _genes = list_to_genes(data, **kwargs)

        elif isinstance(data, dict):
            _genes = dict_to_genes(data, num_of_genes=num_of_genes, **kwargs)

        self.genes: List[Gene] = _genes

        if not self.genes and num_of_genes:
            self.num_of_genes = int(num_of_genes)
        else:
            self.num_of_genes = len(self.genes)

    def __bool__(self) -> bool:
        return bool(self.genes)

    def __len__(self) -> int:
        return len(self.genes)

    def __iter__(self):
        yield from self.genes

    def __getitem__(self, idx):
        return self.genes[idx]

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base for Optimizer that will implemented later on."""

    def __init__():
        pass

    # @abstractmethod
    # def update(self, gen_idx: int, **kwargs):
    # """Implement single update step."""
    # raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs):
        """Implement the logic for the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def validate_operators(self):
        """Validate Operators used in the Optimizer."""
        raise NotImplementedError

    @abstractmethod
    def set_operators(self):
        """Set Operators."""
        raise NotImplementedError


if __name__ == "__main__":
    pass

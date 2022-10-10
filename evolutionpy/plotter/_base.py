"""WIP."""
from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from ..modules._base import BaseOptimizer


class BasePlotter(ABC):
    """Base class for plotting"""

    @abstractmethod
    def _plot(
        self,
        sloutions_to_plot: np.ndarray = None,
        num_genes: int = 0,
        generations_completed: int = 0,
        **kwargs,
    ):
        raise NotImplementedError

    def plot(self, optimizer: BaseOptimizer, save_dir=None, **kwargs):
        """
        Create, shows, and returns a figure.

        Return:
          Figure
        """
        # fig = self._plot(
        #     solutions_to_plot=solutions_to_plot,
        #     num_genes=num_genes,
        #     generations_completed=generations_completed,
        #     **kwargs,
        # )

        # if save_dir:
        #     plt.savefig(fname=save_dir, bbox_inches="tight")
        # return fig
        return self._plot()

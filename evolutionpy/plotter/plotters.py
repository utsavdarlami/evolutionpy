"""WIP."""
from __future__ import annotations

import numpy as np
from loguru import logger

from ._base import BasePlotter


class DefaultPlotter(BasePlotter):
    """Default plotter."""

    def _plot(
        self,
        sloutions_to_plot: np.ndarray = None,
        num_genes: int = 0,
        generations_completed: int = 0,
        **kwargs,
    ):
        return "Default Plot"


class BoxPlotPlotter(BasePlotter):
    def _plot(
        self,
        sloutions_to_plot: np.ndarray = None,
        num_genes: int = 0,
        generations_completed: int = 0,
        **kwargs,
    ):
        return "Box Plot"


class HistogramPlotter(BasePlotter):
    def _plot(
        self,
        sloutions_to_plot: np.ndarray = None,
        num_genes: int = 0,
        generations_completed: int = 0,
        **kwargs,
    ):
        return "Histogram Plot"


class PlotterFactory:

    plotters = {
        "plot": DefaultPlotter,
        "histogram": HistogramPlotter,
        "boxplot": BoxPlotPlotter,
    }

    _default = DefaultPlotter

    def get_plotter(self, method: str = "plot") -> object:
        """Get plotter."""
        if not isinstance(method, str):
            logger.warning(f"Expected str but found {type(method)}")

        return self.plotters.get(method, self._default)()


if __name__ == "__main__":
    pass

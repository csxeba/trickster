from typing import List

import numpy as np
from matplotlib import pyplot as plt, axes

from .history import History


def _plot_smoothed_vector_to_axes(ax: axes.Axes,
                                  vector: np.ndarray,
                                  smoothing_window_size: int,
                                  name: str):

    half_window = smoothing_window_size // 2
    ax.plot(vector[half_window:-half_window], "r-", alpha=0.5)
    ax.plot(np.convolve(vector, np.ones(smoothing_window_size) / smoothing_window_size, mode="valid"))
    ax.set_title(name)
    ax.grid()


def plot_vectors(vectors: List[np.ndarray],
                 names: List[str],
                 smoothing_window_size: int,
                 skip_first: int = 10,
                 **subplots_kwargs):

    figsize = subplots_kwargs.pop("figsize", (16, 9))
    fig, axes = plt.subplots(len(vectors), sharex="all", figsize=figsize, **subplots_kwargs)

    for ax, vec, name in zip(axes, vectors, names):
        vec = vec[skip_first:]
        _plot_smoothed_vector_to_axes(ax, vec, smoothing_window_size, name)

    plt.tight_layout()
    plt.show()


def plot_history(history: History,
                 smoothing_window_size: int,
                 skip_first: int = 10,
                 **subplots_kwargs):

    vectors = [v for k, v in history.gather().items()]
    plot_vectors(vectors, history.keys, smoothing_window_size, skip_first, **subplots_kwargs)

import numpy as np

from matplotlib import pyplot as plt


def plot_vectors(vectors, names, smoothing_window_size, skip_first=10):
    fig, axes = plt.subplots(len(vectors), sharex="all")
    half_window = smoothing_window_size // 2

    for ax, vec, name in zip(axes, vectors, names):
        vec = vec[skip_first:]
        ax.plot(vec[half_window:-half_window], "r-", alpha=0.5)
        ax.plot(np.convolve(vec, np.ones(smoothing_window_size) / smoothing_window_size, mode="valid"))
        ax.set_title(name)
        ax.grid()

    plt.tight_layout()
    plt.show()


def plot_history(history, smoothing_window_size, skip_first=10):
    vectors = [v for k, v in history.gather().items()]
    plot_vectors(vectors, history.keys, smoothing_window_size, skip_first)

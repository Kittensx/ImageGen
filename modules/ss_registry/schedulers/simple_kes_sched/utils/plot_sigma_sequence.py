from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

# Sigma plots are diagnostics written by a background generation worker. Force
# a raster-only backend before importing any plotting classes so Windows never
# opens Figure windows or blocks the worker waiting for a GUI to close.
matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def plot_sigma_sequence(
    sigs: Any,
    stopping_index: int,
    log_filename: str,
    save_directory: str = "image_generation_data",
    show_plot: bool = False,
) -> str:
    """Write one non-interactive sigma-sequence PNG and return its path.

    ``show_plot`` remains in the signature for compatibility with older call
    sites, but is intentionally ignored. Scheduler diagnostics must never open
    an interactive Matplotlib window from the WebUI or CLI worker.
    """

    del show_plot
    directory = Path(save_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(log_filename))[0]
    graph_path = directory / f"{base_filename}_sigma_plot.png"

    if hasattr(sigs, "detach"):
        sigs_np = sigs.detach().cpu().numpy()
    elif hasattr(sigs, "cpu"):
        sigs_np = sigs.cpu().numpy()
    else:
        sigs_np = np.asarray(sigs)
    sigs_np = np.asarray(sigs_np, dtype=float).reshape(-1)
    x = np.arange(len(sigs_np))

    if sigs_np.size:
        stop = max(0, min(int(stopping_index), int(sigs_np.size) - 1))
    else:
        stop = 0

    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x, sigs_np, label="Sigma Sequence", marker="o")
    axis.axvline(
        x=stop,
        color="red",
        linestyle="--",
        label=f"Stopping Point: {stop}",
    )
    axis.set_xlabel("Step Index")
    axis.set_ylabel("Sigma Value")
    axis.set_title("Final Sigma Sequence")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()

    temporary_path = graph_path.with_suffix(graph_path.suffix + ".tmp")
    try:
        figure.savefig(temporary_path, format="png")
        os.replace(temporary_path, graph_path)
    finally:
        figure.clear()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    return str(graph_path)

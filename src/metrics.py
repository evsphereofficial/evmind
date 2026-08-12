"""Forgetting and retention metrics.

F_i (forgetting of task i) = max(best_accuracy_i) - final_accuracy_i,
where best_accuracy_i is the highest accuracy task i achieved at any
evaluation point (normally right after its own training phase).

Also tracked per task: initial accuracy, best historical accuracy, current
accuracy, forgetting. Average forgetting is the mean over all tasks.
"""

from __future__ import annotations

import numpy as np


def compute_forgetting(accuracy_matrix: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute forgetting metrics from an accuracy matrix.

    Args:
        accuracy_matrix: (num_tasks, num_phases) array where entry [i, j] is
            the accuracy (%) of task i measured after training phase j
            (entry is NaN where task i was not yet learned, i.e. j < i).

    Returns:
        Dict with keys: initial, best, final, forgetting (per-task arrays),
        average_forgetting (float), final_average_accuracy (float).
    """
    num_tasks = accuracy_matrix.shape[0]

    initial = np.full(num_tasks, np.nan)
    best = np.full(num_tasks, np.nan)
    final = np.full(num_tasks, np.nan)
    forgetting = np.full(num_tasks, np.nan)

    for i in range(num_tasks):
        row = accuracy_matrix[i, :]
        measured = row[~np.isnan(row)]
        if measured.size == 0:
            continue
        initial[i] = measured[0]          # accuracy right after own training
        best[i] = np.max(measured)        # best historical accuracy
        final[i] = measured[-1]           # accuracy at the very end
        forgetting[i] = best[i] - final[i]

    average_forgetting = float(np.nanmean(forgetting))
    final_average_accuracy = float(np.nanmean(final))

    return {
        "initial": initial,
        "best": best,
        "final": final,
        "forgetting": forgetting,
        "average_forgetting": average_forgetting,
        "final_average_accuracy": final_average_accuracy,
    }
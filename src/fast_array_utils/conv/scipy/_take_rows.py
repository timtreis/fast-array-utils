# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba

from ...numba import njit


if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray


__all__ = ["_compact_rows_csr_numba", "_take_rows_csr_numba"]


@njit
def _take_rows_csr_numba(
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],
    data: NDArray[np.number[Any]],
    rows: NDArray[np.intp],
    new_indptr: NDArray[np.integer[Any]],
    bounds: NDArray[np.intp],
    out_indices: NDArray[np.integer[Any]],
    out_data: NDArray[np.number[Any]],
) -> None:
    # `bounds` splits the output rows into chunks of similar nnz; chunks write disjoint ranges
    for c in numba.prange(bounds.shape[0] - 1):
        for r in range(bounds[c], bounds[c + 1]):
            src = indptr[rows[r]]
            dst = new_indptr[r]
            for t in range(new_indptr[r + 1] - dst):
                out_indices[dst + t] = indices[src + t]
                out_data[dst + t] = data[src + t]


@njit
def _compact_rows_csr_numba(
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],
    data: NDArray[np.number[Any]],
    rows: NDArray[np.intp],
    new_indptr: NDArray[np.integer[Any]],
    windows: NDArray[np.intp],
    scr_indices: NDArray[np.integer[Any]],
    scr_data: NDArray[np.number[Any]],
) -> None:
    # `rows` increase, so every row moves towards the front: a window's writes stay below the rows of
    # later windows. Within a window, gather into scratch first, then write back.
    for w in range(windows.shape[0] - 1):
        a, b = windows[w], windows[w + 1]
        base = new_indptr[a]
        for r in numba.prange(a, b):
            src = indptr[rows[r]]
            dst = new_indptr[r] - base
            for t in range(new_indptr[r + 1] - new_indptr[r]):
                scr_indices[dst + t] = indices[src + t]
                scr_data[dst + t] = data[src + t]
        for t in numba.prange(new_indptr[b] - base):
            indices[base + t] = scr_indices[t]
            data[base + t] = scr_data[t]

# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba

from ...numba import njit


if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray


__all__ = ["_take_rows_csr_numba"]


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


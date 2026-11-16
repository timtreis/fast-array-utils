# SPDX-License-Identifier: MPL-2.0
"""Utilities only for sparse matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray
    from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix

    from ... import types


__all__ = ["compact_rows", "take_rows", "to_dense"]

# value dtypes numba can compile the kernels for
_NUMBA_DTYPES = frozenset(map(np.dtype, ["?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "c8", "c16"]))
# `take_rows` splits the rows into this many chunks of similar nnz, for load balance
_N_CHUNKS = 256
# entries `compact_rows` moves per window; also the size of its scratch buffers
_WINDOW = 1 << 26


def to_dense(x: types.spmatrix | types.sparray, order: Literal["C", "F"] = "C") -> NDArray[Any]:
    """Convert a sparse matrix to a dense matrix.

    Parameters
    ----------
    x
        Input matrix.
    order
        The order of the output matrix.

    Returns
    -------
    Dense matrix form of ``x``

    """
    if TYPE_CHECKING:
        assert isinstance(x, types.CSBase | coo_matrix | coo_array)

    try:
        from ._to_dense import _to_dense_csc_numba, _to_dense_csr_numba
    except ImportError:
        warn("numba is not installed; falling back to slow conversion", RuntimeWarning, stacklevel=2)
        return x.toarray(order=order)

    out = np.zeros(x.shape, dtype=x.dtype, order=order)
    if x.format == "csr":
        _to_dense_csr_numba(x, out)
    elif x.format == "csc":
        _to_dense_csc_numba(x, out)
    else:  # pragma: no cover
        out = x.toarray(order=order)
    return out


def take_rows[M: (csr_matrix, csr_array)](x: M, rows: NDArray[np.integer[Any]] | NDArray[np.bool_] | slice) -> M:
    """Select rows of a CSR matrix, like ``x[rows]``, but in parallel.

    The result is identical to SciPy’s, including its index dtype.

    Parameters
    ----------
    x
        Input matrix.
    rows
        One-dimensional boolean mask, integer indices (any order, negative allowed, repeats allowed), or a slice.

    Returns
    -------
    A new matrix of the same type as ``x`` with the selected rows.

    Raises
    ------
    TypeError
        If ``x`` is not in CSR format.
    IndexError
        If ``rows`` is not one-dimensional, out of range, a mask of the wrong length, or neither integers nor booleans.

    Notes
    -----
    This returns ``x[rows]`` without numba and for value dtypes numba doesn’t support.
    Like every fau kernel, it runs serially, with a warning, where :func:`~fast_array_utils.numba.njit` deems parallel code unsafe.

    See Also
    --------
    compact_rows : Keep increasing rows in place, in ``x``’s own buffers.

    """
    from scipy.sparse import get_index_dtype

    _check_csr(x, "take_rows")
    idx = _row_indices(rows, x.shape[0])  # validates `rows` on every path
    if idx.size == 0 or not _use_numba(x):
        return x[rows]

    from ._take_rows import _take_rows_csr_numba

    lens = x.indptr[idx + 1] - x.indptr[idx]
    nnz = int(lens.sum(dtype=np.int64))
    # as in SciPy’s own indexing; `csr_matrix`’s constructor then narrows int64 to int32 where the values fit
    index_dtype = get_index_dtype((x.indices, x.indptr), maxval=max(nnz, x.shape[1]))
    new_indptr = np.empty(idx.size + 1, dtype=index_dtype)
    new_indptr[0] = 0
    np.cumsum(lens, out=new_indptr[1:])
    bounds = np.unique(np.searchsorted(new_indptr, np.linspace(0, nnz, _N_CHUNKS + 1), side="left"))
    out_indices = np.empty(nnz, dtype=index_dtype)
    out_data = np.empty(nnz, dtype=x.data.dtype)
    _take_rows_csr_numba(x.indptr, x.indices, x.data, idx, new_indptr, bounds, out_indices, out_data)
    # like SciPy’s, `has_sorted_indices` of the result is determined lazily
    return type(x)((out_data, out_indices, new_indptr), shape=(idx.size, x.shape[1]))


def compact_rows(x: csr_matrix | csr_array, rows: NDArray[np.integer[Any]] | NDArray[np.bool_] | slice) -> None:
    """Keep only the selected rows of a CSR matrix, moving them within ``x``’s own buffers.

    Afterwards, ``x`` has the values and indices ``x[rows]`` would have had, but keeps its index dtype.
    With numba, this allocates no second copy of the matrix, only scratch space for about ``2**26`` entries
    and a few arrays with one entry per selected row.
    Without numba, or for value dtypes numba doesn’t support, this computes ``x[rows]`` with SciPy
    and copies it back into ``x``’s buffers, which needs memory for the selected rows,
    so it saves no memory over ``x[rows]``.

    Compaction lowers the peak memory, not the footprint: the buffers keep their size,
    and like after :meth:`~scipy.sparse.csr_array.resize`, ``x`` holds views of their first entries.
    To release the unused tail, copy the buffers, e.g. ``x.data = x.data.copy()``.

    The caller must own ``x``’s buffers: any other matrix or array that shares
    ``x.data``, ``x.indices``, or ``x.indptr`` (e.g. a view, or a matrix built from the same arrays)
    sees the moved entries afterwards. Buffers that are writeable memory maps are compacted on disk.

    Parameters
    ----------
    x
        Input matrix with writeable buffers. Modified in place.
    rows
        Strictly increasing rows: a one-dimensional boolean mask, increasing integer indices, or a slice with a positive step.
        Negative indices are normalised before the check, so ``[-1, 0]`` is not increasing.

    Raises
    ------
    TypeError
        If ``x`` is not in CSR format.
    IndexError
        If ``rows`` is not one-dimensional, out of range, a mask of the wrong length, or neither integers nor booleans.
    ValueError
        If ``rows`` is not strictly increasing, or one of ``x``’s buffers is read-only.

    Notes
    -----
    All checks run before any data moves, so if this raises an :class:`Exception`, ``x`` is unchanged.
    An interrupt (e.g. :class:`KeyboardInterrupt`) while the rows move leaves ``x`` undefined.

    Cached format flags (e.g. ``has_sorted_indices``) are kept, as :meth:`~scipy.sparse.csr_array.resize` keeps them.

    See Also
    --------
    take_rows : Select rows in any order into a new matrix.

    """
    _check_csr(x, "compact_rows")
    n_rows, n_cols = x.shape
    idx = _row_indices(rows, n_rows)
    if not all(a.flags.writeable for a in (x.data, x.indices, x.indptr)):
        msg = "compact_rows needs writeable buffers"
        raise ValueError(msg)
    if idx.size > 1 and (np.diff(idx) <= 0).any():
        msg = "compact_rows needs strictly increasing rows"
        raise ValueError(msg)
    k = idx.size
    if k == n_rows:  # all rows kept: nothing moves
        return

    if _use_numba(x):
        from ._take_rows import _compact_rows_csr_numba

        lens = x.indptr[idx + 1] - x.indptr[idx]
        new_indptr = np.zeros(k + 1, dtype=x.indptr.dtype)
        np.cumsum(lens, out=new_indptr[1:])
        nnz = int(new_indptr[-1])
        # windows of about `_WINDOW` entries; the scratch buffers also fit a window’s last row
        windows = np.unique(np.searchsorted(new_indptr, np.arange(0, nnz + _WINDOW, _WINDOW), side="left").clip(0, k))
        size = min(nnz, _WINDOW + int(lens.max(initial=0)))
        scratch_indices, scratch_data = np.empty(size, x.indices.dtype), np.empty(size, x.data.dtype)
        # fau’s `njit` picks the parallel or the serial kernel, e.g. serial off the main thread
        # unless the threading layer is thread-safe. Both move the rows in place, so the memory bound holds.
        # TODO(cluster): check calls off the main thread. The first one runs `njit`’s threading-layer probe,
        # a subprocess cached per process with a 20 s timeout; measure it and check the layer it reports.
        _compact_rows_csr_numba(x.indptr, x.indices, x.data, idx, new_indptr, windows, scratch_indices, scratch_data)
    else:
        y = x[idx]
        x.data[: y.nnz] = y.data
        x.indices[: y.nnz] = y.indices
        new_indptr = y.indptr
    x.indptr[: k + 1] = new_indptr
    # slices `indptr`, `indices`, and `data` to views of their first entries
    x.resize((k, n_cols))


def _check_csr(x: csr_matrix | csr_array, name: str) -> None:
    if x.format != "csr":
        msg = f"{name} needs a CSR matrix, got {x.format!r}"
        raise TypeError(msg)


def _use_numba(x: csr_matrix | csr_array) -> bool:
    """Whether to use the numba kernels for ``x``; otherwise SciPy is used.

    Threads and forks need no check here: :func:`~fast_array_utils.numba.njit` runs the serial kernel where parallel code is unsafe.
    """
    from importlib.util import find_spec

    return x.data.dtype in _NUMBA_DTYPES and find_spec("numba") is not None


def _row_indices(rows: NDArray[np.integer[Any]] | NDArray[np.bool_] | slice, n_rows: int) -> NDArray[np.intp]:
    """Return non-negative row indices for ``rows``, validated like SciPy does."""
    if isinstance(rows, slice):
        return np.arange(n_rows)[rows]
    rows = np.asarray(rows)
    if rows.ndim != 1:
        msg = f"row index must be one-dimensional, got {rows.ndim} dimensions"
        raise IndexError(msg)
    if rows.dtype == np.bool_:
        if rows.shape != (n_rows,):
            msg = f"boolean index of shape {rows.shape} does not match {n_rows} rows"
            raise IndexError(msg)
        return np.flatnonzero(rows)
    if rows.dtype.kind not in "iu":
        msg = f"row indices must be integers or booleans, got {rows.dtype}"
        raise IndexError(msg)
    # checked before the cast to intp, which could wrap uint64
    if rows.size and (rows.min() < -n_rows or rows.max() >= n_rows):
        msg = f"row index out of range for {n_rows} rows"
        raise IndexError(msg)
    idx = rows.astype(np.intp, copy=False)
    return np.where(idx < 0, idx + n_rows, idx)

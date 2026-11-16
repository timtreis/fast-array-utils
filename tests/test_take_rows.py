# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sp

from fast_array_utils.conv.scipy import take_rows
from testing.fast_array_utils import SUPPORTED_TYPES


if TYPE_CHECKING:
    from typing import Any

    from pytest_codspeed import BenchmarkFixture

    from testing.fast_array_utils import ArrayType


# take_rows takes SciPy CSR matrices and arrays only
ATS_CSR = {at for at in SUPPORTED_TYPES if at.mod == "scipy.sparse" and at.name.startswith("csr_")}
csr_types = pytest.mark.array_type(ATS_CSR)
needs_numba = pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")


def _assert_identical(actual: sp.csr_matrix | sp.csr_array, expected: sp.csr_matrix | sp.csr_array) -> None:
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape
    for attr in ("indptr", "indices", "data"):
        a, e = getattr(actual, attr), getattr(expected, attr)
        assert a.dtype == e.dtype, attr
        np.testing.assert_array_equal(a, e, strict=True)
    assert actual.has_sorted_indices == expected.has_sorted_indices


@pytest.fixture
def x(array_type: ArrayType[sp.csr_matrix | sp.csr_array, None]) -> sp.csr_matrix | sp.csr_array:
    return array_type.random((500, 40), density=0.1, dtype=np.float32)


@pytest.fixture(params=[True, False], ids=["numba", "scipy"])
def use_numba(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Run with the numba kernels (if installed) and with the SciPy fallback."""
    from fast_array_utils.conv import scipy as conv

    if request.param:
        if not find_spec("numba"):
            pytest.skip("numba not installed")
    else:
        monkeypatch.setattr(conv, "_use_numba", lambda _: False)
    return bool(request.param)


_RNG = np.random.default_rng(1)


@pytest.mark.usefixtures("use_numba")
@csr_types
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(_RNG.random(500) > 0.3, id="mask"),
        pytest.param(np.arange(0, 500, 3), id="sorted"),
        pytest.param(_RNG.permutation(500)[:200], id="unsorted"),
        pytest.param(np.array([-1, -500, 3, 3, 499]), id="negative-repeats"),
        pytest.param(np.array([7], dtype=np.int32), id="int32"),
        pytest.param(slice(10, 400, 2), id="slice"),
        pytest.param(slice(None, None, -1), id="slice-reversed"),
        pytest.param(np.zeros(500, dtype=bool), id="empty-mask"),
        pytest.param(np.array([], dtype=np.int64), id="empty-indices"),
    ],
)
def test_take_rows(x: sp.csr_matrix | sp.csr_array, rows: Any) -> None:
    _assert_identical(take_rows(x, rows), x[rows])


def test_take_rows_unsorted_indices() -> None:
    x = sp.csr_matrix((np.array([1.0, 2.0, 3.0]), np.array([2, 0, 1]), np.array([0, 2, 3])), shape=(2, 3))
    assert not x.has_sorted_indices
    _assert_identical(take_rows(x, np.array([1, 0])), x[np.array([1, 0])])


@csr_types
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("rows", [pytest.param(np.arange(0, 500, 2), id="indices"), pytest.param(slice(5, 5), id="empty-slice")])
def test_take_rows_index_dtype(x: sp.csr_matrix | sp.csr_array, index_dtype: type[np.signedinteger[Any]], rows: Any) -> None:
    x.indices, x.indptr = x.indices.astype(index_dtype), x.indptr.astype(index_dtype)
    _assert_identical(take_rows(x, rows), x[rows])


@pytest.mark.skipif(not os.environ.get("FAU_TEST_LARGE"), reason="needs ~40 GB of memory; set FAU_TEST_LARGE=1")
def test_take_rows_index_dtype_promotion() -> None:
    """Selecting more nnz than int32 holds promotes to int64, as SciPy does."""
    n_per_row = 2**16
    x = sp.csr_matrix(
        (np.ones(n_per_row, dtype=np.int8), np.arange(n_per_row, dtype=np.int32), np.array([0, n_per_row], dtype=np.int32)),
        shape=(1, n_per_row),
    )
    rows = np.zeros(2**15 + 1, dtype=np.intp)  # 2**31 + 2**16 nnz
    actual = take_rows(x, rows)  # SciPy’s own `x[rows]` fails here (int32 overflow)
    assert actual.indptr.dtype == actual.indices.dtype == np.int64
    np.testing.assert_array_equal(actual.indptr, np.arange(rows.size + 1) * n_per_row)
    np.testing.assert_array_equal(actual.indices[-n_per_row:], x.indices)
    assert actual.nnz == rows.size * n_per_row


@csr_types
@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble], ids=["longdouble", "clongdouble"])
def test_take_rows_unsupported_dtype_falls_back(x: sp.csr_matrix | sp.csr_array, dtype: type) -> None:
    x = x.astype(dtype)
    _assert_identical(take_rows(x, np.array([2, 0])), x[np.array([2, 0])])


@needs_numba
@csr_types
def test_take_rows_off_main_thread(x: sp.csr_matrix | sp.csr_array) -> None:
    """Off the main thread, fau’s `njit` picks the kernel (it may warn and run serially)."""
    from concurrent.futures import ThreadPoolExecutor

    rows = _RNG.permutation(500)[:100]
    with ThreadPoolExecutor(1) as pool:
        # TODO(cluster): the first call off the main thread runs `njit`’s threading-layer probe (timeout 20 s)
        actual = pool.submit(take_rows, x, rows).result(timeout=60)
    _assert_identical(actual, x[rows])


@csr_types
def test_take_rows_empty_int64_array(x: sp.csr_matrix | sp.csr_array) -> None:
    x.indices, x.indptr = x.indices.astype(np.int64), x.indptr.astype(np.int64)
    rows = np.array([], dtype=np.int64)
    _assert_identical(take_rows(x, rows), x[rows])


def test_take_rows_uint64() -> None:
    x = sp.csr_matrix(np.eye(4))
    _assert_identical(take_rows(x, np.array([3, 1], dtype=np.uint64)), x[np.array([3, 1], dtype=np.uint64)])
    with pytest.raises(IndexError):
        take_rows(x, np.array([2**63], dtype=np.uint64))  # must not wrap to a negative index


@pytest.mark.usefixtures("use_numba")
@csr_types
@pytest.mark.parametrize("rows", [np.array([500]), np.array([-501]), np.ones(499, dtype=bool)])
def test_take_rows_errors(x: sp.csr_matrix | sp.csr_array, rows: Any) -> None:
    with pytest.raises(IndexError):
        take_rows(x, rows)
    with pytest.raises(IndexError):
        x[rows]


_NOT_1D = [
    pytest.param(np.array(3), id="0d-int"),
    pytest.param(np.array(1, dtype=bool), id="0d-bool"),
    pytest.param(np.array([[0, 1], [2, 3]]), id="2d-int"),
    pytest.param(np.ones((500, 1), dtype=bool), id="2d-bool"),
]


@pytest.mark.usefixtures("use_numba")
@csr_types
@pytest.mark.parametrize("rows", _NOT_1D)
def test_take_rows_not_1d(x: sp.csr_matrix | sp.csr_array, rows: Any) -> None:
    """Every path rejects indices that aren’t one-dimensional, so the result can’t depend on the path."""
    with pytest.raises(IndexError, match="one-dimensional"):
        take_rows(x, rows)


def test_take_rows_rejects_csc() -> None:
    with pytest.raises(TypeError, match="CSR"):
        take_rows(sp.csc_matrix((3, 3)), np.array([0]))  # type: ignore[type-var]


@pytest.fixture(scope="module")
def bench_x(array_type: ArrayType[sp.csr_matrix | sp.csr_array, None]) -> sp.csr_matrix | sp.csr_array:
    return array_type.random((100_000, 2_000), density=0.05, dtype=np.float32)


@pytest.mark.benchmark
@needs_numba
@csr_types
def test_take_rows_benchmark(benchmark: BenchmarkFixture, bench_x: sp.csr_matrix | sp.csr_array) -> None:
    rows = np.random.default_rng(1).random(bench_x.shape[0]) > 0.1
    take_rows(bench_x, rows)  # warmup: numba compile
    benchmark(take_rows, bench_x, rows)


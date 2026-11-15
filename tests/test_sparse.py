# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import os
from contextlib import nullcontext
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils.conv.scipy import to_dense
from testing.fast_array_utils import Flags


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.types import CSBase
    from testing.fast_array_utils import ArrayType
    from testing.fast_array_utils._array_type import _DTypeLikeNum


pytestmark = [pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")]


WARNS_NUMBA = pytest.warns(RuntimeWarning, match="numba is not installed; falling back to slow conversion")


@pytest.fixture(scope="session", params=["csr", "csc"])
def sp_fmt(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return cast('Literal["csr", "csc"]', request.param)


@pytest.fixture(scope="session", params=["array", "matrix"])
def sp_container(request: pytest.FixtureRequest) -> Literal["array", "matrix"]:
    return cast('Literal["array", "matrix"]', request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> type[np.float32 | np.float64]:
    return cast("type[np.float32 | np.float64]", request.param)


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense(array_type: ArrayType[CSBase, None], order: Literal["C", "F"], dtype: _DTypeLikeNum) -> None:
    mat = array_type.random((10, 10), density=0.1, dtype=dtype)
    with WARNS_NUMBA if not find_spec("numba") else nullcontext():
        arr = to_dense(mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == mat.dtype
    np.testing.assert_equal(arr, mat.toarray(order=order))


@pytest.mark.benchmark
@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")
def test_to_dense_benchmark(
    benchmark: BenchmarkFixture,
    array_type: ArrayType[CSBase, None],
    order: Literal["C", "F"],
    dtype: _DTypeLikeNum,
) -> None:
    mat = array_type.random((10_000, 10_000), dtype=dtype)
    to_dense(mat, order=order)  # warmup: numba compile
    benchmark(to_dense, mat, order=order)


@pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")
def test_numba_types_cacheable(tmp_path: Path) -> None:
    """Sparse numba types pickle by reference, so cached kernels taking them load in a new session."""
    import pickle
    import subprocess
    import sys

    import numba
    import scipy.sparse as sp

    from fast_array_utils._plugins.numba_sparse import TYPES

    for cls in TYPES:  # each class is bound under its own name
        assert pickle.loads(pickle.dumps(cls)) is cls
    typ = numba.typeof(sp.random(3, 3, density=0.5, format="csr"))
    assert pickle.loads(pickle.dumps(typ)) == typ

    (tmp_path / "kernel_mod.py").write_text(
        "import numba, numpy as np, scipy.sparse as sp, fast_array_utils\n"
        "@numba.njit(cache=True)\n"
        "def nnz(x):\n    return x.indptr[-1]\n"
        "nnz(sp.random(5, 5, density=0.5, format='csr', dtype=np.float32))\n"
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache"), "NUMBA_DEBUG_CACHE": "1", "PYTHONPATH": str(tmp_path)}
    runs = [subprocess.run([sys.executable, "-c", "import kernel_mod"], env=env, capture_output=True, text=True, check=True).stdout for _ in range(2)]
    assert "data saved" in runs[0]
    assert "data loaded" in runs[1]
    assert "data saved" not in runs[1]

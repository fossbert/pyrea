"""Results returned from R must not point into R memory (use-after-free: values changed later)."""

import gc

import numpy as np
import pandas as pd
import pytest

import pyrea as pr


def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


@pytest.fixture
def data(rng):
    genes = [f"g{i}" for i in range(400)]
    samples = [f"s{i}" for i in range(40)]
    expr = pd.DataFrame(rng.normal(size=(400, 40)), index=genes, columns=samples)
    sets = {f"TF{i}": genes[i * 30:(i + 1) * 30] for i in range(8)}
    return expr, pr.gene_sets_to_regulon(sets, minsize=20)


def _owns_memory(a):
    """True if the numpy buffer behind ``a`` belongs to numpy (not to an R object)."""
    b = a
    while isinstance(b, np.ndarray):
        if b.flags.owndata:
            return True
        b = b.base
    return False


def _stress_r(n=25):
    """Make R allocate, collect garbage and reuse memory."""
    import rpy2.robjects as ro

    for _ in range(n):
        ro.r("x <- numeric(3e6); x[1] <- 1; rm(x); invisible(gc())")
    gc.collect()


@viper
def test_viper_activity_survives_later_r_calls(data):
    expr, reg = data
    result = pr.viper_activity(expr, reg, method="none", minsize=20)
    snapshot = result.to_numpy().copy()
    for _ in range(4):
        pr.viper_activity(expr, reg, method="mad", minsize=20)
    _stress_r()
    np.testing.assert_array_equal(result.to_numpy(), snapshot)
    assert not np.isnan(result.to_numpy()).any()
    assert _owns_memory(result.to_numpy())


@viper
def test_area_survives_later_r_calls(data):
    expr, reg = data
    result = pr.aREA(expr, reg, minsize=20)
    snapshot = result.to_numpy().copy()
    for _ in range(4):
        pr.aREA(expr, reg, minsize=20)
    _stress_r()
    np.testing.assert_array_equal(result.to_numpy(), snapshot)
    assert _owns_memory(result.to_numpy())


@viper
def test_read_regulon_survives_later_r_calls(tmp_path, data):
    import rpy2.robjects as ro
    from pyrea.rbackend.regulon import regulon_to_r

    _, reg = data
    path = tmp_path / "reg.rds"
    ro.r["saveRDS"](regulon_to_r(reg), str(path))
    back = pr.read_regulon_rds(path)
    snapshot = back[["mor", "likelihood"]].to_numpy().copy()
    _stress_r()
    np.testing.assert_array_equal(back[["mor", "likelihood"]].to_numpy(), snapshot)
    assert _owns_memory(back["mor"].to_numpy()) and _owns_memory(back["likelihood"].to_numpy())

"""pyrea.rbackend.area.aREA -- thin wrapper around viper::aREA (needs R)."""

import numpy as np
import pandas as pd
import pytest

from pyrea.rbackend.area import aREA
from pyrea.utils import gene_sets_to_regulon


def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


@pytest.fixture
def regulon(rng):
    genes = [f"g{i}" for i in range(200)]
    genesets = {
        "TF1": list(rng.choice(genes, 40, replace=False)),
        "TF2": list(rng.choice(genes, 35, replace=False)),
    }
    reg = gene_sets_to_regulon(genesets, minsize=20)
    reg["mor"] = rng.choice([-1, 1], size=len(reg)) * rng.uniform(0.3, 1, len(reg))
    reg["likelihood"] = rng.uniform(0.2, 1, len(reg))
    return reg, genes


@viper
def test_area_dataframe(rng, regulon):
    reg, genes = regulon
    dset = pd.DataFrame(rng.normal(size=(len(genes), 4)), index=genes, columns=list("abcd"))

    nes = aREA(dset, reg, minsize=20)

    assert isinstance(nes, pd.DataFrame)
    assert set(nes.index) == {"TF1", "TF2"}
    assert list(nes.columns) == list("abcd")
    assert np.isfinite(nes.to_numpy()).all()


@viper
def test_area_series_uses_its_name_as_the_only_column(rng, regulon):
    reg, genes = regulon
    sig = pd.Series(rng.normal(size=len(genes)), index=genes, name="my_signature")

    nes = aREA(sig, reg, minsize=20)

    assert list(nes.columns) == ["my_signature"]


@viper
def test_area_dset_filter_matches_unfiltered(rng, regulon):
    reg, genes = regulon
    dset = pd.DataFrame(rng.normal(size=(len(genes), 3)), index=genes, columns=list("abc"))

    nes = aREA(dset, reg, minsize=20)
    nes_filtered = aREA(dset, reg, minsize=20, dset_filter=True)

    # filtering to regulon targets before ranking changes the ranks, so results
    # differ, but both should score the same regulators/samples.
    assert nes.shape == nes_filtered.shape
    assert not nes.equals(nes_filtered)


@viper
def test_area_raises_when_no_regulon_meets_minsize(regulon):
    from rpy2.rinterface_lib.embedded import RRuntimeError

    reg, genes = regulon
    dset = pd.Series(np.arange(5.0), index=genes[:5])  # far fewer genes than minsize

    with pytest.raises(RRuntimeError):
        aREA(dset, reg, minsize=20)

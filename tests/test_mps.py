"""pyrea.mps -- mutant phenotype score."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyrea as pr
from pyrea.mps import lr_to_rl


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
    """GENE_A: mutants have low G, normal RPT. GENE_B: mutants have low RPT only. GENE_C: no signal."""
    n = 80
    samples = [f"s{i}" for i in range(n)]
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    mut = pd.DataFrame(0, index=genes, columns=samples)
    for g in genes:
        mut.loc[g, rng.choice(samples, 20, replace=False)] = 1
    G = pd.DataFrame(rng.normal(size=(3, n)), index=genes, columns=samples)
    R = pd.DataFrame(rng.normal(size=(3, n)), index=genes, columns=samples)
    G.loc["GENE_A", mut.loc["GENE_A"] == 1] -= 3
    R.loc["GENE_B", mut.loc["GENE_B"] == 1] -= 3
    return mut, G, R


def test_lr_to_rl():
    assert lr_to_rl(3) == pytest.approx(0.5)
    assert lr_to_rl(1) == 0


def test_rl_is_antisymmetric_and_bounded(rng):
    a, b = rng.normal(0, 1, 50), rng.normal(2, 1, 50)
    x = np.linspace(-3, 5, 40)
    rl = pr.relative_likelihood(x, a, b)
    assert np.all(np.abs(rl) <= 1)
    np.testing.assert_allclose(rl, -pr.relative_likelihood(x, b, a))
    assert rl[0] > 0 > rl[-1]  # mutant-like on the left, WT-like on the right


def test_rl_nan_for_degenerate_groups():
    x = np.arange(5.0)
    assert np.isnan(pr.relative_likelihood(x, [1, 1, 1], [0, 1, 2, 3])).all()
    assert np.isnan(pr.relative_likelihood(x, [1.0], [0, 1, 2, 3])).all()


def test_mps_separates_and_uses_best_trait(data):
    mut, G, R = data
    mps = pr.mutant_phenotype_score(mut, G, R)
    assert set(mps.index) == {"GENE_A", "GENE_B", "GENE_C"}
    for g in ("GENE_A", "GENE_B"):  # signal in G for A, in RPT for B -- both picked up
        m = mut.loc[g] == 1
        assert mps.loc[g, m].mean() > 0.5 > -0.5 > mps.loc[g, ~m].mean()
    # G alone cannot see GENE_B
    g_only = pr.mutant_phenotype_score(mut, G)
    assert g_only.loc["GENE_B"].abs().mean() < mps.loc["GENE_B"].abs().mean()


def test_mps_thresholds_and_missing_genes(data):
    mut, G, R = data
    mps = pr.mutant_phenotype_score(mut, G.drop("GENE_C"), R, min_mut=21)
    assert mps.empty
    assert "GENE_C" not in pr.mutant_phenotype_score(mut, G.drop("GENE_C"), R).index


def test_mps_aligns_samples(data):
    mut, G, R = data
    mps = pr.mutant_phenotype_score(mut, G.iloc[:, ::-1], R.iloc[:, :60])
    assert list(mps.columns) == list(mut.columns.intersection(R.columns[:60], sort=False))


def test_classify_phenotype():
    s = pd.Series([0.9, 0.5, 0.1, -0.6, np.nan], index=list("abcde"))
    out = pr.classify_phenotype(s)
    assert list(out[:4]) == ["mutant", "intermediate", "intermediate", "wt"]
    assert pd.isna(out["e"])
    assert pr.classify_phenotype(s, lr=1.5)["c"] == "intermediate"  # t = 0.2


def test_plots_run(data):
    mut, G, R = data
    mps = pr.mutant_phenotype_score(mut, G, R)
    ax = pr.plot_mps_rank(mps.loc["GENE_A"], mut.loc["GENE_A"])
    assert ax.get_ylabel() == "MPS"
    assert pr.plot_rl_diagnostic("GENE_A", mut, G, R) is not None


@viper
def test_mutation_association_direction(data):
    mut, G, R = data
    res = pr.mutation_association(mut, {"G": G, "RPT": R}, min_mut=10)
    assert res.loc["GENE_A", "nes_G"] < -3
    assert res.loc["GENE_B", "nes_RPT"] < -3
    assert res.loc["GENE_A", "trait_min"] == "G"
    assert res.loc["GENE_B", "trait_min"] == "RPT"
    assert res.loc["GENE_C", "p_min"] > res.loc["GENE_A", "p_min"]


@viper
def test_locus_specific(rng, data):
    mut, G, R = data
    # split GENE_A carriers into a functional and a passenger variant
    carriers = list(mut.columns[mut.loc["GENE_A"] == 1])
    v = pd.DataFrame(0, index=["GENE_A:p.X1Y", "GENE_A:p.Z2*"], columns=mut.columns)
    v.loc["GENE_A:p.X1Y", carriers[:12]] = 1
    v.loc["GENE_A:p.Z2*", carriers[12:]] = 1
    # make the second variant a passenger: normal G
    G.loc["GENE_A", carriers[12:]] = rng.normal(size=len(carriers) - 12)
    assoc, mps = pr.locus_specific_mps(v, G, R, min_samples=2, min_mps=10, min_wt=10)
    assert assoc.loc["GENE_A:p.X1Y", "p_min"] < assoc.loc["GENE_A:p.Z2*", "p_min"]
    assert list(mps.index) == ["GENE_A:p.X1Y"]  # passenger has only 8 carriers < min_mps
    m = v.loc["GENE_A:p.X1Y"] == 1
    assert mps.loc["GENE_A:p.X1Y", m].mean() > 0.5


def test_mps_nan_excludes_not_profiled(data):
    mut, G, R = data
    m = mut.astype(float)
    nanned = list(m.columns[(m.loc["GENE_A"] == 0).to_numpy()][:15])
    m.loc["GENE_A", nanned] = np.nan
    # reference: same data, those samples removed from the groups but still scored
    full = pr.mutant_phenotype_score(mut, G, R, min_wt=1)
    res = pr.mutant_phenotype_score(m, G, R, min_wt=1)
    assert res.shape == full.shape and res.loc["GENE_A"].notna().all()
    assert not np.allclose(res.loc["GENE_A"], full.loc["GENE_A"])  # WT density changed
    # NaN must not be treated as WT: with too few real WT samples the gene is skipped
    assert "GENE_A" not in pr.mutant_phenotype_score(m, G, R, min_wt=(mut.loc["GENE_A"] == 0).sum() - 14)


@viper
def test_association_ignores_not_profiled(data):
    mut, G, R = data
    m = mut.astype(float)
    wt_cols = list(m.columns[(m.loc["GENE_A"] == 0).to_numpy()])
    m.loc["GENE_A", wt_cols[:20]] = np.nan
    res = pr.mutation_association(m, {"G": G}, min_mut=10)
    assert res.loc["GENE_A", "n_mut"] == 20 and res.loc["GENE_A", "nes_G"] < -3


def test_mps_key_maps_rows_to_genes(data):
    mut, G, R = data
    fus = mut.loc[["GENE_A"]].rename(index={"GENE_A": "A-B fusion"})
    res = pr.mutant_phenotype_score(fus, G, R, key={"A-B fusion": "GENE_A"})
    ref = pr.mutant_phenotype_score(mut.loc[["GENE_A"]], G, R)
    assert list(res.index) == ["A-B fusion"]
    np.testing.assert_allclose(res.iloc[0], ref.iloc[0])
    assert pr.mutant_phenotype_score(fus, G, R).empty        # unmapped row is no gene

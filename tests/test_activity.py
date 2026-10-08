"""viper_activity, viper_rpt, read_regulon_rds."""

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
def expr_and_regulon(rng):
    genes = [f"g{i}" for i in range(300)]
    samples = [f"s{i}" for i in range(30)]
    expr = pd.DataFrame(rng.normal(size=(300, 30)), index=genes, columns=samples)
    sets = {"TF1": genes[:40], "TF2": genes[40:80], "TF3": genes[80:120]}
    reg = pr.gene_sets_to_regulon(sets, minsize=20)
    expr.loc[sets["TF1"], samples[:10]] += 2.0  # TF1 targets up in first 10 samples
    return expr, reg


@pytest.mark.parametrize("method", ["rank", "lineal"])
def test_rpt_residuals_orthogonal_to_expression(rng, method):
    expr = pd.DataFrame(rng.normal(size=(5, 40)), index=list("abcde"), columns=range(40))
    act = 0.8 * expr + pd.DataFrame(rng.normal(size=(5, 40)), index=expr.index, columns=expr.columns)
    rpt = pr.viper_rpt(act, expr, method=method)
    assert rpt.shape == act.shape
    xs = expr.rank(axis=1) if method == "rank" else expr
    corr = [np.corrcoef(rpt.loc[g], xs.loc[g])[0, 1] for g in rpt.index]
    assert np.abs(corr).max() < 1e-8
    np.testing.assert_allclose(rpt.mean(axis=1), 0, atol=1e-8)


def test_rpt_aligns_genes_and_samples(rng):
    expr = pd.DataFrame(rng.normal(size=(4, 20)), index=list("abcd"), columns=range(20))
    act = pd.DataFrame(rng.normal(size=(3, 15)), index=list("bcx"), columns=range(5, 20))
    rpt = pr.viper_rpt(act, expr)
    assert list(rpt.index) == ["b", "c"] and list(rpt.columns) == list(range(5, 20))


def test_rpt_rejects_unknown_method(rng):
    d = pd.DataFrame(rng.normal(size=(2, 5)))
    with pytest.raises(ValueError):
        pr.viper_rpt(d, d, method="spline")


@viper
@pytest.mark.parametrize("method", ["rank", "lineal"])
def test_rpt_matches_viper_viperRPT(rng, method):
    from pyrea.rbackend import r_package
    from pyrea.rbackend.regulon import frame_to_r_matrix, matrix_from_r

    expr = pd.DataFrame(rng.normal(size=(6, 25)), index=list("abcdef"), columns=[f"s{i}" for i in range(25)])
    expr.iloc[0, :5] = expr.iloc[0, 5]  # ties
    act = pd.DataFrame(rng.normal(size=(6, 25)), index=expr.index, columns=expr.columns) + expr * 0.5
    ref = matrix_from_r(r_package("viper").viperRPT(frame_to_r_matrix(act), frame_to_r_matrix(expr),
                                                    method=method))
    mine = pr.viper_rpt(act, expr, method=method)
    np.testing.assert_allclose(mine.loc[ref.index, ref.columns].to_numpy(), ref.to_numpy(), atol=1e-8)


@viper
def test_viper_activity_detects_activation(expr_and_regulon):
    expr, reg = expr_and_regulon
    vp = pr.viper_activity(expr, reg, minsize=20)
    assert set(vp.index) == {"TF1", "TF2", "TF3"}
    assert list(vp.columns) == list(expr.columns)
    assert vp.loc["TF1", expr.columns[:10]].mean() > vp.loc["TF1", expr.columns[10:]].mean() + 2
    assert abs(vp.loc["TF2"].mean()) < 1


@viper
def test_pipeline_activity_rpt_mps(rng, expr_and_regulon):
    expr, reg = expr_and_regulon
    tf_rows = pd.DataFrame(rng.normal(size=(3, 30)), index=["TF1", "TF2", "TF3"], columns=expr.columns)
    expr = pd.concat([expr, tf_rows])
    vp = pr.viper_activity(expr, reg, minsize=20)
    rpt = pr.viper_rpt(vp, expr)
    assert list(rpt.index) == ["TF1", "TF2", "TF3"]
    mut = pd.DataFrame(0, index=["TF1"], columns=expr.columns)
    mut.loc["TF1", expr.columns[:10]] = 1
    mps = pr.mutant_phenotype_score(mut, vp, rpt, min_mut=5, min_wt=5)
    assert mps.loc["TF1", expr.columns[:10]].mean() > mps.loc["TF1", expr.columns[10:]].mean()


@viper
def test_regulon_rds_roundtrip(tmp_path, expr_and_regulon):
    from pyrea.rbackend import require_rpy2
    from pyrea.rbackend.regulon import regulon_to_r

    ro, *_ = require_rpy2()
    _, reg = expr_and_regulon
    path = tmp_path / "reg.rds"
    ro.r["saveRDS"](regulon_to_r(reg), str(path))
    back = pr.read_regulon_rds(path)
    cols = ["source", "target", "mor", "likelihood"]
    pd.testing.assert_frame_equal(
        back[cols].sort_values(cols[:2]).reset_index(drop=True),
        reg[cols].sort_values(cols[:2]).reset_index(drop=True),
        check_dtype=False,
    )

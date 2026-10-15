"""pyrea.comutation: effect of a co-alteration with and without a driver."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pyrea as pr


def _data(rng, n=300, bD=1.0, bC=0.5, bDC=1.2, noise=1.0, pD=0.5, pC=0.5, extra=None):
    s = [f"s{i}" for i in range(n)]
    D = pd.Series((rng.random(n) < pD).astype(float), index=s, name="D")
    C = pd.Series((rng.random(n) < pC).astype(float), index=s, name="C")
    y = noise * rng.normal(size=n) + bD * D + bC * C + bDC * D * C
    if extra is not None:
        y = y + extra
    act = pd.DataFrame([y], index=["T"], columns=s)
    return D, C, act


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_recovers_effects_and_interaction(rng):
    D, C, act = _data(rng)
    res = pr.comutation_effect(D, C, "T", act, n_boot=300, n_perm=500)
    assert res.cell_names == ("neither", "D only", "C only", "D+C")
    assert res.cells["n"].sum() == 300 and (res.cells["n"] > 40).all()
    m = res.model
    assert m.loc[("G", "D:C"), "low"] < 1.2 < m.loc[("G", "D:C"), "high"]                  # true interaction inside the interval
    assert m.loc[("G", "D:C"), "p_perm"] < 0.01 and m.loc[("G", "D:C"), "p_hc3"] < 0.001
    assert m.loc[("G", "D"), "beta"] == pytest.approx(1.0, abs=0.4)
    e = res.effects.loc["G"]
    assert e.loc["interaction C by D", "low"] > 0 and e.loc["interaction D by C", "low"] > 0
    assert e.loc["C within D", "delta"] > e.loc["C within not-D", "delta"]
    assert e.loc["C within D", "n1"] == res.cells.loc["D+C", "n"] and e.loc["C within D", "n2"] == res.cells.loc["D only", "n"]
    assert (e["low"].dropna() <= e["delta"].dropna() + 1e-9).all() and (e["delta"].dropna() <= e["high"].dropna() + 1e-9).all()
    assert "min detectable" in repr(res) and res.min_detectable_interaction > 0


def test_main_effect_of_partner_is_not_an_interaction(rng):
    D, C, act = _data(rng, bC=1.0, bDC=0.0, n=400)
    res = pr.comutation_effect(D, C, "T", act, n_boot=300, n_perm=500)
    assert res.model.loc[("G", "C"), "p_perm"] < 0.01                                    # C acts, with or without D
    assert res.model.loc[("G", "D:C"), "p_perm"] > 0.05
    assert res.effects.loc[("G", "interaction C by D"), "low"] < 0 < res.effects.loc[("G", "interaction C by D"), "high"]


def test_permutation_p_is_roughly_uniform_under_the_null(rng):
    ps = []
    for i in range(60):
        D, C, act = _data(np.random.default_rng(100 + i), n=80, bC=0.0, bD=0.0, bDC=0.0)
        r = pr.comutation_effect(D, C, "T", act, n_boot=0, n_perm=200, seed=i)
        ps.append(r.model.loc[("G", "D:C"), "p_perm"])
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.15 and 0.35 < ps.mean() < 0.65


def test_freedman_lane_agrees_with_robust_p_for_large_effects(rng):
    D, C, act = _data(rng, n=250, bDC=1.5)
    m = pr.comutation_effect(D, C, "T", act, n_boot=0, n_perm=800).model
    assert abs(m.loc[("G", "D"), "p_perm"] - m.loc[("G", "D"), "p_hc3"]) < 0.02 or m.loc[("G", "D"), "p_perm"] < 0.01
    assert m["p_perm"].between(1 / 801, 1).all()


def test_covariates_remove_confounding(rng):
    n = 400
    s = [f"s{i}" for i in range(n)]
    sub = pd.Series(rng.choice(["a", "b"], n), index=s, name="subtype")
    C = pd.Series(np.where(sub == "b", rng.random(n) < 0.7, rng.random(n) < 0.15).astype(float), index=s, name="C")
    D = pd.Series((rng.random(n) < 0.5).astype(float), index=s, name="D")
    y = rng.normal(size=n) + 1.5 * (sub == "b") + 1.0 * D                              # C has no effect of its own
    act = pd.DataFrame([y], index=["T"], columns=s)
    raw = pr.comutation_effect(D, C, "T", act, n_boot=200, n_perm=200)
    adj = pr.comutation_effect(D, C, "T", act, covariates=sub, n_boot=200, n_perm=200)
    assert raw.model.loc[("G", "C"), "p_perm"] < 0.01                                  # C looks effective through the subtype
    assert abs(adj.model.loc[("G", "C"), "beta"]) < 0.4 < raw.model.loc[("G", "C"), "beta"]     # coefficient shrinks to ~0 (CI coverage: 95 % over seeds)
    assert adj.covariates == ("subtype",) and "subtype" in adj.scores.columns
    d_raw = raw.effects.loc[("G", "C within not-D"), "delta"]
    d_adj = adj.effects.loc[("G", "C within not-D"), "delta"]
    d_unadj = pr.comutation_effect(D, C, "T", act, covariates=sub, adjusted=False, n_boot=0, n_perm=0).effects.loc[("G", "C within not-D"), "delta"]
    assert abs(d_adj) < abs(d_raw) - 0.15 and d_unadj == pytest.approx(d_raw)           # rank effects adjusted by default


def test_small_cells_warn_and_give_nan(rng):
    D, C, act = _data(rng, n=120)
    C[(D == 1)] = 0.0
    C[(D == 1)[(D == 1)].index[:2]] = 1.0                                              # only two D+C samples
    with pytest.warns(UserWarning, match="fewer than min_cell"):
        res = pr.comutation_effect(D, C, "T", act, n_boot=50, n_perm=50)
    e = res.effects.loc["G"]
    assert np.isnan(e.loc["C within D", "delta"]) and np.isnan(e.loc["interaction C by D", "delta"])
    assert np.isfinite(e.loc["D within not-C", "delta"])
    assert res.cells.loc["D+C", "n"] == 2
    empty_c = C.copy()
    empty_c[:] = 0.0
    with pytest.warns(UserWarning):
        r = pr.comutation_effect(D, empty_c, "T", act, n_boot=20, n_perm=20)
    assert np.isnan(r.min_detectable_interaction) and r.model["beta"].isna().all()


def test_input_handling(rng):
    D, C, act = _data(rng, n=150)
    D2 = D.copy()
    D2.iloc[:10] = np.nan
    res = pr.comutation_effect(D2, C, "T", act, n_boot=0, n_perm=0)
    assert res.cells["n"].sum() == 140                                                  # not profiled: excluded
    with pytest.raises(ValueError, match="no activity"):
        pr.comutation_effect(D, C, "NOPE", act)
    with pytest.raises(ValueError, match="different names"):
        pr.comutation_effect(D, D, "T", act)
    renamed = pr.comutation_effect(D, C, "T", act, names=("RHOA", "TP53"), n_boot=0, n_perm=0)
    assert renamed.cell_names == ("neither", "RHOA only", "TP53 only", "RHOA+TP53") and ("G", "RHOA:TP53") in renamed.model.index
    cov = pd.Series(rng.choice(["x", "y", None], 150), index=D.index)
    r = pr.comutation_effect(D, C, "T", act, covariates=cov, n_boot=0, n_perm=0)
    assert r.n_dropped == int(cov.isna().sum()) and r.cells["n"].sum() == 150 - r.n_dropped
    num = pr.comutation_effect(D, C, "T", act, covariates=pd.DataFrame({"purity": rng.random(150), "cohort": rng.choice(["a", "b"], 150)}, index=D.index),
                               n_boot=0, n_perm=0)
    assert num.covariates == ("purity", "cohort")
    a = pr.comutation_effect(D, C, "T", act, n_boot=100, n_perm=100, seed=3)
    b = pr.comutation_effect(D, C, "T", act, n_boot=100, n_perm=100, seed=3)
    pd.testing.assert_frame_equal(a.effects, b.effects), pd.testing.assert_frame_equal(a.model, b.model)
    off = pr.comutation_effect(D, C, "T", act, n_boot=0, n_perm=0)
    assert off.effects["low"].isna().all() and off.model["p_perm"].isna().all() and off.model["p_hc3"].notna().all()


def test_several_scores_and_detectable_effect(rng):
    D, C, act = _data(rng, n=200)
    expr = pd.DataFrame([rng.normal(size=200)], index=["T"], columns=D.index)
    rpt = pd.DataFrame([rng.normal(size=200)], index=["T"], columns=D.index)
    res = pr.comutation_effect(D, C, "T", act, rpt=rpt, expression=expr, n_boot=50, n_perm=50)
    assert set(res.effects.index.get_level_values("score")) == {"G", "RPT", "expr"} and set(res.model.index.get_level_values("score")) == {"G", "RPT", "expr"}
    assert {"mean_G", "median_RPT", "mean_expr"} <= set(res.cells.columns)
    small = pr.comutation_effect(D.iloc[:60], C.iloc[:60], "T", act.iloc[:, :60], n_boot=0, n_perm=0)
    assert small.min_detectable_interaction > res.min_detectable_interaction          # fewer samples: larger detectable effect


def test_figures_axes_and_statistics(rng):
    D, C, act = _data(rng, n=200)
    res = pr.comutation_effect(D, C, "T", act, expression=pd.DataFrame([rng.normal(size=200)], index=["T"], columns=D.index),
                               n_boot=60, n_perm=60)
    fig = res.plot()
    assert len([a for a in fig.axes if a.get_title(loc="left")]) == 3
    assert all(len(a.texts) <= 1 for a in fig.axes)                                      # clean: only the panel letter
    full = res.plot(stats=True)
    assert sum(len(a.texts) for a in full.axes) >= sum(len(a.texts) for a in fig.axes) + 2
    bracket = [t.get_text() for a in full.axes for t in a.texts if t.get_text().startswith("delta")]
    assert bracket and not bracket[0].startswith("delta -")                              # D+C vs. D only: positive here
    _, axs = plt.subplots(1, 3)
    out = res.plot(axes=axs)
    assert [o is a for o, a in zip(out, axs)] == [True] * 3
    with pytest.raises(ValueError, match="Need 3 axes"):
        res.plot(axes=axs[:2])
    ax = res.plot_cells("G")
    assert len(ax.texts) == 0 and [t.get_text().split("\n")[0] for t in ax.get_xticklabels()] == ["neither", "D only", "C only", "D+C"]
    assert res.add_cell_stats("D only", "neither", "G", ax=ax) is ax and len(ax.texts) == 1
    with pytest.raises(ValueError, match="Unknown cell"):
        res.add_cell_stats("D only", "nope", "G", ax=ax)
    with pytest.raises(ValueError, match="Unknown score"):
        res.plot_cells("nope")
    ia = res.plot_interaction("G")
    assert len(ia.lines) >= 2 and len(ia.texts) == 0
    assert "interaction D:C" in res.add_interaction_stats("G", ax=ia).texts[0].get_text()
    fx = res.plot_effects(scores=["G"], contrasts=["C within D", "interaction C by D"])
    assert [t.get_text() for t in fx.get_yticklabels()] == ["C within D | G", "interaction: C by D | G"]
    res.add_effect_p(ax=fx)
    twin = [a for a in fx.figure.axes if a is not fx][0]
    assert [t.get_text().startswith("p") for t in twin.get_yticklabels()] == [True, False]
    with pytest.raises(ValueError, match="plot_effects"):
        res.add_effect_p(ax=res.plot_cells("G"))
    with pytest.raises(ValueError, match="Unknown score"):
        res.plot_effects(scores=["nope"])


def test_collinear_covariates_are_dropped_with_a_warning(rng):
    D, C, act = _data(rng, n=240)
    cohort = pd.Series(rng.choice(["stad", "esca"], 240), index=D.index)
    subtype = pd.Series(np.where(cohort == "esca", rng.choice(["E_a", "E_b"], 240), rng.choice(["S_a", "S_b", "S_c"], 240)), index=D.index)
    cov = pd.DataFrame({"cohort": cohort, "subtype": subtype})
    with pytest.warns(UserWarning, match="linear combinations"):
        res = pr.comutation_effect(D, C, "T", act, covariates=cov, n_boot=0, n_perm=50)
    assert res.model["beta"].notna().all()                                              # the model is fitted on the remaining columns
    alone = pr.comutation_effect(D, C, "T", act, covariates=subtype.to_frame("subtype"), n_boot=0, n_perm=0)
    assert res.model.loc[("G", "D:C"), "beta"] == pytest.approx(alone.model.loc[("G", "D:C"), "beta"], abs=0.1)


def test_covariate_level_with_one_sample_gives_finite_intervals(rng):
    D, C, act = _data(rng, n=200)
    sub = pd.Series(rng.choice(["a", "b"], 200), index=D.index)
    sub.iloc[0] = "rare"                                                                 # a level with a single sample (leverage 1)
    res = pr.comutation_effect(D, C, "T", act, covariates=sub, n_boot=0, n_perm=0)
    assert np.isfinite(res.model[["se", "low", "high", "p_hc3"]].to_numpy()).all()

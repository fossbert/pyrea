"""pyrea.activity_specificity_across_cohorts: activity of regulator sets in many cohorts (VIPER replaced by a simple stand-in)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pyrea as pr
from pyrea.activity_specificity_multi import _meta


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


CALLS = []


def fake_viper(expression, regulon, method="mad", minsize=25, eset_filter=True, pleiotropy=False, nes=True, cores=1, pleiotropy_args=None):
    """Stand-in for viper: mean z-score of the targets. With ``pleiotropy`` the targets shared with another regulon are dropped."""
    CALLS.append(bool(pleiotropy))
    z = expression.sub(expression.mean(axis=1), axis=0).div(expression.std(axis=1), axis=0)
    reg = regulon[regulon["target"].isin(z.index)]
    if pleiotropy:
        reg = reg[reg.groupby("target")["source"].transform("size") == 1]
    rows = {}
    for tf, d in reg.groupby("source"):
        if len(d) >= minsize:
            rows[tf] = (z.loc[d["target"]].mul(d["mor"].to_numpy(), axis=0)).mean(axis=0)
    return pd.DataFrame(rows).T


def fail_viper(expression, regulon, **kw):
    if kw.get("pleiotropy"):
        raise RuntimeError("Fehler in regind[, 2:1] : falsche Anzahl von Dimensionen")
    return fake_viper(expression, regulon, **kw)


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr("pyrea.rbackend.viper.viper_activity", fake_viper)


def _cohort(seed, n=90, n_genes=600, drop=("",)):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(n_genes)]
    S, U = rng.normal(size=(2, n))
    X = rng.normal(size=(n_genes, n))
    X[:200] += S                                    # stroma factor on genes 0-199
    X[30:60] += 1.5 * U                             # the shared targets of A and B
    ids = [f"TCGA.A{seed}.{i:04d}.01A.11R.A1.31" for i in range(n)]
    expr = pd.DataFrame(X, index=genes, columns=ids)
    rows = []
    def add(tf, idx):
        rows.extend((tf, genes[i], 1.0, 1.0) for i in idx)
    add("A", range(0, 60))
    add("B", range(30, 90))
    add("C", range(100, 160))
    add("D", range(300, 360))
    for k in range(40):
        add(f"bg{k}", rng.choice(n_genes, 50, replace=False))
    reg = pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"]).drop_duplicates(["source", "target"])
    reg = reg[~reg["source"].isin(drop)]
    purity = pd.Series(S + 0.3 * rng.normal(size=n), index=[pr.tcga_sample_id(c) for c in ids], name="purity")
    return reg, expr, purity


@pytest.fixture(scope="module")
def cohorts():
    out, pur = {}, []
    for i, name in enumerate(["BRCA", "STAD", "GBM"]):
        reg, expr, p = _cohort(i)
        out[name] = (reg, expr)
        pur.append(p)
    return out, pd.concat(pur)


SETS = [("A", "B"), ("A", "C")]


def test_pipeline_tables_summary_meta_and_figures(cohorts):
    co, purity = cohorts
    purity = purity.copy()
    purity.iloc[:10] = np.nan                                                        # 10 BRCA samples without purity
    ma = pr.activity_specificity_across_cohorts(co, SETS, covariates=purity, sample_id=pr.tcga_sample_id, minsize=20, verbose=False)
    assert isinstance(ma, pr.MultiActivity) and "MultiActivity(3 cohort(s)" in repr(ma)
    cov = ma.coverage.set_index("cohort")
    assert cov.loc["BRCA", "n_samples"] == 80 and cov.loc["BRCA", "n_dropped"] == 10 and cov.loc["STAD", "n_dropped"] == 0
    assert (cov["pleiotropy"] == "ok").all() and (cov["n_sets"] == 2).all()
    assert ma.table.columns[0] == "cohort" and len(ma.table) == 6 and set(ma.pairs["variant"]) == {"full", "pleiotropy"}
    s = ma.summary().set_index("set")
    assert s.loc["A | B", "n_cohorts"] == 3 and s.loc["A | B", "n_top"] == 3 and s.loc["A | C", "n_top"] == 0
    assert s.loc["A | B", "median_rho"] > s.loc["A | C", "median_rho"] and s.index[0] == "A | B"
    m = ma.meta().set_index(["tf1", "tf2", "adjusted"])
    ab, ab_adj = m.loc[("A", "B", False)], m.loc[("A", "B", True)]
    assert ab["k"] == 3 and 0.5 < ab["rho_pooled"] < 1 and ab["ci_low"] < ab["rho_pooled"] < ab["ci_high"] and ab["p"] < 1e-6
    assert ab_adj["rho_pooled"] < ab["rho_pooled"] and ab_adj["n_total"] == 80 + 90 + 90 and 0 <= ab["I2"] <= 1
    assert list(ma.meta(adjusted=True)["adjusted"].unique()) == [True]
    pm = ma.pair_matrix("partial_rho")
    assert pm.shape == (3, 2) and list(pm.columns) == ["A | B", "A | C"] and list(pm.index) == ["BRCA", "STAD", "GBM"]
    assert (pm["A | B"] > pm["A | C"]).all()
    pe = ma.pleiotropy_effect()
    assert len(pe) == 6 and (pe[pe["tf2"] == "B"]["change"] < 0).all() and {"partial_rho_full", "partial_rho_pleiotropy"} <= set(pe.columns)
    mat = ma.matrix()
    assert mat.shape == (3, 2) and (mat["A | B"] < 0.05).all()
    fig, ax = plt.subplots()
    assert ma.plot_matrix(ax=ax) is ax
    n0 = len(ax.texts)
    ma.add_top(ax=ax)
    assert len(ax.texts) - n0 == 3
    fig2, ax2 = plt.subplots()
    assert ma.plot_forest("A", "B", adjusted=True, ax=ax2) is ax2
    assert "pooled" in [t.get_text() for t in ax2.get_yticklabels()] and "BRCA" in [t.get_text() for t in ax2.get_yticklabels()]
    with pytest.raises(ValueError, match="Draw the matrix first"):
        ma.add_top(ax=plt.subplots()[1])
    with pytest.raises(ValueError, match="not evaluated"):
        ma.plot_forest("A", "D")


def test_without_covariates_and_without_pleiotropy(cohorts):
    co, _ = cohorts
    ma = pr.activity_specificity_across_cohorts(co, SETS, pleiotropy=False, minsize=20, verbose=False)
    assert set(ma.pairs["variant"]) == {"full"} and "partial_rho" not in ma.pairs.columns and ma.settings["covariates"] == []
    assert (ma.coverage["pleiotropy"] == "off").all() and True not in CALLS
    assert list(ma.summary().columns)[:3] == ["set", "n_cohorts", "median_rho"]
    with pytest.raises(ValueError, match="no covariates"):
        ma.meta(adjusted=True)
    with pytest.raises(ValueError, match="No pairs for variant"):
        ma.meta(variant="pleiotropy")
    with pytest.raises(ValueError, match="no pleiotropy variant"):
        ma.pleiotropy_effect()
    assert ma.matrix("padj_ext_percentile_max").shape == (3, 2)                    # falls back to the unadjusted percentile
    with pytest.raises(ValueError, match="Unknown value"):
        ma.matrix("nope")
    with pytest.raises(ValueError, match="Unknown value"):
        ma.pair_matrix("nope")


def test_pleiotropy_failure_is_recorded_not_raised(cohorts, monkeypatch):
    co, _ = cohorts
    monkeypatch.setattr("pyrea.rbackend.viper.viper_activity", fail_viper)
    ma = pr.activity_specificity_across_cohorts(co, SETS, minsize=20, verbose=False)
    assert ma.coverage["pleiotropy"].str.startswith("failed: ").all() and set(ma.pairs["variant"]) == {"full"}


def test_missing_regulators_skipped_and_duplicated_samples(cohorts):
    co, purity = cohorts
    co = dict(co)
    co["STAD"] = (co["STAD"][0][~co["STAD"][0]["source"].isin(["C"])], co["STAD"][1])
    co["none"] = (co["GBM"][0][co["GBM"][0]["source"].str.startswith("bg")], co["GBM"][1])
    with pytest.warns(UserWarning, match="skipped"):
        ma = pr.activity_specificity_across_cohorts(co, SETS, minsize=20, verbose=False)
    cov = ma.coverage.set_index("cohort")
    assert cov.loc["STAD", "n_sets"] == 1 and cov.loc["STAD", "missing"] == "C" and set(ma.skipped) == {"none"}
    assert ma.summary().set_index("set").loc["A | C", "n_cohorts"] == 2
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="None of the cohorts"):
        pr.activity_specificity_across_cohorts({"none": co["none"]}, SETS, minsize=20, verbose=False)
    short = lambda c: pr.tcga_sample_id(c, 12)                                       # patient level: the vials repeat -> dedupe
    reg, expr, _ = _cohort(0)
    expr2 = expr.copy()
    expr2.columns = [c if i % 2 == 0 else expr.columns[i - 1] for i, c in enumerate(expr.columns)]
    ma2 = pr.activity_specificity_across_cohorts({"x": (reg, expr2)}, [("A", "B")], sample_id=pr.tcga_sample_id, pleiotropy=False, minsize=20, verbose=False)
    assert ma2.coverage["n_duplicated_samples"].iloc[0] == 45 and ma2.coverage["n_samples"].iloc[0] == 45
    assert short("TCGA.AA.0001.01A.11R.A1.31") == "TCGA-AA-0001"


def test_cache_and_merge_by_name(cohorts, tmp_path):
    co, purity = cohorts
    files = {}
    for name, (reg, expr) in co.items():
        f = tmp_path / f"{name}.tsv"
        expr.to_csv(f, sep="\t")
        files[name] = (reg, f)
    kw = dict(covariates=purity, sample_id=pr.tcga_sample_id, minsize=20, cache_dir=tmp_path / "cache", verbose=False)
    a = pr.activity_specificity_across_cohorts(files, SETS, **kw)
    n_calls = len(CALLS)
    b = pr.activity_specificity_across_cohorts(files, SETS, **kw)
    assert len(CALLS) == n_calls                                                      # served from the cache
    pd.testing.assert_frame_equal(a.table, b.table)
    pr.activity_specificity_across_cohorts(files, SETS, **{**kw, "direction": "positive"})
    assert len(CALLS) > n_calls                                                       # other settings: recomputed
    n_calls = len(CALLS)
    pr.activity_specificity_across_cohorts(files, SETS, **{**kw, "covariates": purity * 2})
    assert len(CALLS) > n_calls                                                       # other covariates: recomputed

    nocache = {**kw, "cache_dir": None}
    joint = pr.activity_specificity_across_cohorts(co, SETS, **nocache)
    p1 = pr.activity_specificity_across_cohorts({"BRCA": co["BRCA"]}, SETS, **nocache)
    p2 = pr.activity_specificity_across_cohorts({"STAD": co["STAD"], "GBM": co["GBM"]}, SETS, **nocache)
    mg = p1.merge(p2)
    for attr in ("table", "pairs", "extension", "coverage"):
        pd.testing.assert_frame_equal(getattr(mg, attr), getattr(joint, attr))
    assert list(mg.results) == ["BRCA", "STAD", "GBM"] and mg.settings["n_cohorts"] == 3
    with pytest.raises(ValueError, match="more than one"):
        p1.merge(pr.activity_specificity_across_cohorts({"BRCA": co["STAD"]}, SETS, **nocache))
    with pytest.raises(ValueError, match="differs in"):
        p1.merge(pr.activity_specificity_across_cohorts({"STAD": co["STAD"]}, SETS, **{**nocache, "direction": "positive"}))
    with pytest.raises(ValueError, match="Nothing"):
        pr.MultiActivity.concat([])


def test_export_and_validation(cohorts, tmp_path):
    co, purity = cohorts
    ma = pr.activity_specificity_across_cohorts(co, SETS, covariates=purity, sample_id=pr.tcga_sample_id, minsize=20, verbose=False)
    out = ma.export(tmp_path / "act.xlsx")
    sheets = set(pd.ExcelFile(out).sheet_names)
    assert {"settings", "coverage", "summary", "meta", "meta_adjusted", "table", "pairs", "extension"} <= sheets
    assert (ma.export(tmp_path / "csv") / "meta.csv").exists()
    with pytest.raises(ValueError, match="non-empty dict"):
        pr.activity_specificity_across_cohorts({}, SETS)
    with pytest.raises(ValueError, match="network, expression"):
        pr.activity_specificity_across_cohorts({"x": co["BRCA"][0]}, SETS)
    with pytest.raises(ValueError, match="duplicated"):
        pr.activity_specificity_across_cohorts(co, SETS, covariates=pd.concat([purity, purity.iloc[:2]]))
    with pytest.raises(ValueError, match="picklable"):
        pr.activity_specificity_across_cohorts(co, SETS, sample_id=lambda c: c, n_jobs=2)
    ok = pr.activity_specificity_across_cohorts({"x": {"network": co["BRCA"][0], "expression": co["BRCA"][1]}}, SETS, pleiotropy=False,
                                                minsize=20, verbose=False)
    assert list(ok.results) == ["x"]


def test_meta_analysis_matches_the_formulas():
    one = _meta(np.array([0.5]), np.array([103.0]), 0)
    z, se = np.arctanh(0.5), np.sqrt(1.06 / 100)
    assert one["k"] == 1 and one["rho_pooled"] == pytest.approx(0.5) and one["ci_low"] == pytest.approx(np.tanh(z - 1.96 * se))
    same = _meta(np.array([0.4, 0.4, 0.4]), np.array([100.0, 200.0, 50.0]), 0)
    assert same["I2"] == 0 and same["tau2"] == 0 and same["rho_pooled"] == pytest.approx(0.4)
    het = _meta(np.array([0.1, 0.5, 0.9]), np.array([300.0, 300.0, 300.0]), 0)
    assert het["I2"] > 0.8 and het["p_heterogeneity"] < 0.01 and het["tau2"] > 0
    wide = _meta(np.array([0.1, 0.5, 0.9]), np.array([300.0, 300.0, 300.0]), 0)
    fixed_like = _meta(np.array([0.45, 0.5, 0.55]), np.array([300.0, 300.0, 300.0]), 0)
    assert wide["ci_high"] - wide["ci_low"] > fixed_like["ci_high"] - fixed_like["ci_low"]       # heterogeneity widens the interval
    assert _meta(np.array([np.nan]), np.array([50.0]), 0)["k"] == 0
    assert _meta(np.array([0.3]), np.array([5.0]), 3)["k"] == 0                                   # no degrees of freedom left
    adj = _meta(np.array([0.5]), np.array([103.0]), 3)
    assert adj["ci_high"] - adj["ci_low"] > one["ci_high"] - one["ci_low"]                         # covariates cost degrees of freedom


def test_registry_and_gene_list_covariates_in_the_pipeline(cohorts, tmp_path, monkeypatch):
    monkeypatch.setenv("PYREA_DATA", str(tmp_path / "pyrea_data"))
    co, purity = cohorts
    pr.add_covariates("mine", pd.DataFrame({"my_purity": purity.to_numpy()}, index=purity.index))
    kw = dict(sample_id=pr.tcga_sample_id, minsize=20, pleiotropy=False, verbose=False)
    a = pr.activity_specificity_across_cohorts(co, SETS, covariates=purity, **kw)
    b = pr.activity_specificity_across_cohorts(co, SETS, covariates="my_purity", **kw)
    for attr in ("pairs", "table", "extension"):
        pd.testing.assert_frame_equal(getattr(a, attr), getattr(b, attr))                  # a registry name = the same values as a Series
    assert a.settings["covariates"] == ["purity"] and b.settings["covariates"] == ["my_purity"]
    stroma = [f"g{i}" for i in range(200)]
    c = pr.activity_specificity_across_cohorts(co, SETS, covariates={"Stroma": stroma}, **kw)
    assert c.settings["covariates"] == ["Stroma"] and (c.coverage["n_dropped"] == 0).all()
    ac = c.pairs.query("variant == 'full' and tf2 == 'C'")
    assert (ac["partial_rho"] < ac["rho"] - 0.2).all()                                      # the stroma score explains the A | C link
    d = pr.activity_specificity_across_cohorts(co, SETS, covariates=["my_purity", {"Stroma": stroma}], **kw)
    assert d.settings["covariates"] == ["my_purity", "Stroma"] and d.meta(adjusted=True)["k"].eq(3).all()
    with pytest.raises(ValueError, match="sample_id=pr.tcga_sample_id"):
        pr.activity_specificity_across_cohorts(co, SETS, covariates="absolute_purity", sample_id=None, minsize=20, pleiotropy=False, verbose=False)
    with pytest.raises(ValueError, match="registry name"):
        pr.activity_specificity_across_cohorts(co, SETS, covariates=3.14, **kw)
    # the cache follows the covariate specification
    cache = tmp_path / "cache"
    pr.activity_specificity_across_cohorts(co, SETS, covariates={"Stroma": stroma}, cache_dir=cache, **kw)
    n = len(CALLS)
    pr.activity_specificity_across_cohorts(co, SETS, covariates={"Stroma": stroma}, cache_dir=cache, **kw)
    assert len(CALLS) == n
    pr.activity_specificity_across_cohorts(co, SETS, covariates={"Stroma": stroma[:150]}, cache_dir=cache, **kw)
    assert len(CALLS) > n
    n = len(CALLS)
    pr.activity_specificity_across_cohorts(co, SETS, covariates={"Stroma": stroma}, cache_dir=cache, **{**kw, "minsize": 21})
    assert len(CALLS) > n


# ---------------------------------------------------------------------- with R
def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


def _overlapping(rng, n=70):
    """Eight regulons that overlap their neighbours (40 and 20 targets), every one with its own factor in the expression (viper's
    pleiotropy correction needs several significant, overlapping regulons in a sample)."""
    genes = [f"g{i}" for i in range(600)]
    X = pd.DataFrame(rng.normal(size=(600, n)), index=genes, columns=[f"s{i}" for i in range(n)])
    sets = {f"TF{i}": genes[i * 20:i * 20 + 60] for i in range(8)}
    for i in range(8):
        X.iloc[i * 20:i * 20 + 60] += 0.8 * rng.normal(size=n)
    return X, pr.gene_sets_to_regulon(sets, minsize=20)


@viper
def test_viper_pleiotropy_args(monkeypatch, rng):
    monkeypatch.undo()                                                                   # the real viper here
    X, reg = _overlapping(rng)
    plain = pr.viper_activity(X, reg, method="mad", minsize=20, eset_filter=False)
    pl = pr.viper_activity(X, reg, method="mad", minsize=20, eset_filter=False, pleiotropy=True)
    pl2 = pr.viper_activity(X, reg, method="mad", minsize=20, eset_filter=False, pleiotropy=True,
                            pleiotropy_args={"regulators": 0.5, "shadow": 0.5, "targets": 5, "penalty": 10, "method": "absolute"})
    assert plain.shape == pl.shape == pl2.shape and np.isfinite(pl.to_numpy()).all()
    assert not np.allclose(plain.to_numpy(), pl.to_numpy())                              # the correction changes the activities
    assert not np.allclose(pl.to_numpy(), pl2.to_numpy())                                # and the arguments reach viper
    with pytest.raises(ValueError, match="Unknown pleiotropy_args"):
        pr.viper_activity(X, reg, minsize=20, pleiotropy=True, pleiotropy_args={"nope": 1})
    with pytest.raises(ValueError, match="adaptive"):
        pr.viper_activity(X, reg, minsize=20, pleiotropy=True, pleiotropy_args={"method": "x"})
    ignored = pr.viper_activity(X, reg, method="mad", minsize=20, eset_filter=False, pleiotropy_args={"nope": 1})   # no pleiotropy: ignored
    np.testing.assert_allclose(ignored.to_numpy(), plain.to_numpy())


@viper
def test_cohort_pipeline_with_the_real_viper(monkeypatch, rng, tmp_path):
    monkeypatch.undo()
    X, reg = _overlapping(rng)
    f = tmp_path / "expr.tsv"
    X.to_csv(f, sep="\t")
    ma = pr.activity_specificity_across_cohorts({"c1": (reg, f), "c2": (reg, X)}, [("TF0", "TF1"), ("TF0", "TF1", "TF2")], minsize=20,
                                                min_samples=20, n_jobs=2, verbose=False)
    assert list(ma.results) == ["c1", "c2"] and len(ma.table) == 4
    assert (ma.coverage["pleiotropy"].isin(["ok"]) | ma.coverage["pleiotropy"].str.startswith("failed")).all()
    assert ma.pairs.query("variant == 'full'")["rho"].notna().all()
    np.testing.assert_allclose(ma.table.query("cohort == 'c1'")["rho_mean"].to_numpy(), ma.table.query("cohort == 'c2'")["rho_mean"].to_numpy())


def test_a_worker_tolerates_settings_of_an_older_caller(cohorts):
    """Spawned workers import the current module while a long-running kernel may still hold an older caller: missing newer settings use defaults."""
    from pyrea.activity_specificity_multi import _run_cohort
    co, purity = cohorts
    kw = dict(method="mad", minsize=20, direction="abs", top=5, min_samples=20, pleiotropy=False, pleiotropy_args=None)        # no score_minsize
    name, out, info = _run_cohort("BRCA", co["BRCA"], [("A", "B")], kw, purity, pr.tcga_sample_id, None, None, None)
    assert out is not None and info["n_sets"] == 1 and out.settings["covariates"] == ["purity"]

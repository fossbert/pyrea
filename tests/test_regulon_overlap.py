"""pyrea.regulon_overlap: overlap of regulons by tail."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pyrea as pr


def _regulon(rng, n_genes=2000, n_background=60, planted=40, discordant=0):
    """A and B share `planted` positive targets; discordant: A + / B - ; many background regulators of random size."""
    genes = [f"g{i}" for i in range(n_genes)]
    rows = []
    def add(tf, idx, signs):
        for g, s in zip(idx, signs):
            rows.append((tf, genes[g], s, rng.uniform(0.2, 1.0)))
    shared = rng.choice(n_genes, planted, replace=False)
    rest = np.setdiff1d(np.arange(n_genes), shared)
    a_own = rng.choice(rest, 120, replace=False)
    b_own = rng.choice(np.setdiff1d(rest, a_own), 100, replace=False)
    a_idx = np.concatenate([shared, a_own])
    add("A", a_idx, np.where(rng.random(len(a_idx)) < 0.85, 1.0, -1.0))
    b_signs = np.where(rng.random(planted + 100) < 0.85, 1.0, -1.0)
    b_signs[:planted] = 1.0
    add("B", np.concatenate([shared, b_own]), b_signs)
    c_idx = rng.choice(rest, 110, replace=False)
    add("C", c_idx, np.where(rng.random(110) < 0.85, 1.0, -1.0))
    if discordant:
        d_idx = rng.choice(a_idx[planted:], discordant, replace=False)           # D: negative where A is positive
        add("D", d_idx, -np.ones(discordant))
        add("D", rng.choice(rest, 80, replace=False), np.ones(80))
    for k in range(n_background):
        idx = rng.choice(n_genes, int(rng.integers(30, 200)), replace=False)
        add(f"bg{k}", idx, np.where(rng.random(len(idx)) < 0.8, 1.0, -1.0))
    reg = pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"])
    return reg.drop_duplicates(["source", "target"])


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_planted_overlap_and_tails(rng):
    reg = _regulon(rng)
    pos = pr.regulon_overlap(reg, ["A", "B", "C"], n_perm=300)
    assert pos.tail == "positive" and list(pos.table["sign1"].unique()) == ["+"] and len(pos.table) == 3
    ab = pos.table[(pos.table.tf1 == "A") & (pos.table.tf2 == "B")].iloc[0]
    assert ab["overlap"] >= 30 and ab["fold"] > 3 and ab["p_hyper"] < 1e-6 and ab["p_perm"] < 0.01
    ac = pos.table[(pos.table.tf1 == "A") & (pos.table.tf2 == "C")].iloc[0]
    assert ac["p_hyper"] > 0.01 and ac["p_perm"] > 0.01                                   # no planted overlap
    neg = pr.regulon_overlap(reg, ["A", "B"], tail="negative")
    assert list(neg.table[["sign1", "sign2"]].iloc[0]) == ["-", "-"]
    both = pr.regulon_overlap(reg, ["A", "B"], tail="both")
    assert sorted(zip(both.table.sign1, both.table.sign2)) == [("+", "+"), ("+", "-"), ("-", "+"), ("-", "-")]
    assert "p_perm" not in both.table.columns and both.n_perm == 0
    # the table rows of positive and both agree for the + / + combination
    r1 = pos.table[(pos.table.tf1 == "A") & (pos.table.tf2 == "B")].iloc[0]
    r2 = both.table[(both.table.sign1 == "+") & (both.table.sign2 == "+")].iloc[0]
    assert r1["overlap"] == r2["overlap"] and r1["p_hyper"] == pytest.approx(r2["p_hyper"])


def test_counts_expected_and_hypergeometric_by_hand(rng):
    reg = pd.DataFrame({"source": ["A"] * 4 + ["B"] * 3 + ["bg"] * 3,
                        "target": ["g1", "g2", "g3", "g4", "g3", "g4", "g5", "g6", "g7", "g8"],
                        "mor": [1, 1, 1, -1, 1, -1, 1, 1, 1, 1.0], "likelihood": 1.0})
    ov = pr.regulon_overlap(reg, ["A", "B"], tail="both")
    assert ov.universe_size == 8                                                           # g1..g8
    t = ov.table.set_index(["sign1", "sign2"])
    assert t.loc[("+", "+"), "n1"] == 3 and t.loc[("+", "+"), "n2"] == 2 and t.loc[("+", "+"), "overlap"] == 1   # g3
    assert t.loc[("-", "-"), "overlap"] == 1 and t.loc[("+", "-"), "overlap"] == 0 and t.loc[("-", "+"), "overlap"] == 0
    assert t.loc[("+", "+"), "expected"] == pytest.approx(3 * 2 / 8)
    assert t.loc[("+", "+"), "jaccard"] == pytest.approx(1 / 4) and t.loc[("+", "+"), "overlap_coef"] == pytest.approx(1 / 2)
    from scipy.stats import hypergeom
    assert t.loc[("+", "+"), "p_hyper"] == pytest.approx(hypergeom.sf(0, 8, 3, 2))
    p = ov.pairs.iloc[0]
    assert (p["shared"], p["concordant"], p["discordant"], p["agreement"]) == (2, 2, 0, 1.0)


def test_sign_threshold_and_likelihood_filter():
    reg = pd.DataFrame({"source": ["A"] * 4 + ["B"] * 4, "target": ["g1", "g2", "g3", "g4"] * 2,
                        "mor": [0.1, 0.5, -0.1, -0.6, 0.05, 0.9, 0.0, -0.8], "likelihood": [1.0, 1.0, 1.0, 0.1] * 2})
    base = pr.regulon_overlap(reg, ["A", "B"], tail="both")
    assert base.sizes.loc["B", "n_positive"] == 2 and base.sizes.loc["B", "n_negative"] == 1          # mor == 0 is neither
    strict = pr.regulon_overlap(reg, ["A", "B"], tail="both", mor_min=0.3)
    assert strict.sizes.loc["A", "n_positive"] == 1 and strict.sizes.loc["A", "n_negative"] == 1
    lik = pr.regulon_overlap(reg, ["A", "B"], tail="both", min_likelihood=0.5)
    assert lik.sizes.loc["A", "n_targets"] == 3 and "g4" not in lik.targets.index
    assert lik.universe_size == base.universe_size == 4                                              # the filter does not shrink the universe


def test_degree_weighted_null_corrects_for_hubs(rng):
    """Regulators that share hub targets overlap more than the hypergeometric null expects; the degree-weighted null knows."""
    n = 1500
    genes = np.arange(n)
    hubs = genes[:100]
    rows = []
    for tf in [f"bg{i}" for i in range(80)]:                                                 # background regulators prefer hubs
        sel = np.unique(np.concatenate([rng.choice(hubs, 60), rng.choice(genes, 40)]))
        rows += [(tf, f"g{g}", 1.0, 1.0) for g in sel]
    for tf in ("A", "B"):
        sel = np.unique(np.concatenate([rng.choice(hubs, 60), rng.choice(genes, 40)]))       # no direct relation between A and B
        rows += [(tf, f"g{g}", 1.0, 1.0) for g in sel]
    reg = pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"])
    ov = pr.regulon_overlap(reg, ["A", "B"], n_perm=500, seed=1)
    r = ov.table.iloc[0]
    assert r["overlap"] > r["expected"] * 2 and r["p_hyper"] < 1e-6                         # looks highly significant ...
    assert r["perm_mean"] > r["expected"] * 2 and r["p_perm"] > 0.01                          # ... but is explained by the hubs


def test_permutation_p_is_roughly_uniform_without_overlap():
    ps = []
    for i in range(40):
        reg = _regulon(np.random.default_rng(300 + i), planted=0, n_background=30)
        ov = pr.regulon_overlap(reg, ["A", "C"], n_perm=150, seed=i)
        ps.append(ov.table["p_perm"].iloc[0])
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.15 and 0.3 < ps.mean() < 0.7


def test_shared_matrix_and_discordance(rng):
    reg = _regulon(rng, discordant=25)
    ov = pr.regulon_overlap(reg, ["A", "B", "D"], tail="both")
    sh = ov.shared("A", "D")
    assert (~sh["concordant"]).sum() >= 20 and set(sh.columns) >= {"mor1", "mor2", "likelihood1", "likelihood2", "sign1", "sign2", "concordant"}
    assert len(ov.shared("A", "D", signs=("+", "-"))) == (sh["sign1"].eq("+") & sh["sign2"].eq("-")).sum()
    assert ov.pairs.set_index(["tf1", "tf2"]).loc[("A", "D"), "discordant"] >= 20
    mat = ov.matrix("overlap", ("+", "-"))
    assert mat.loc["A", "D"] == ov.table[(ov.table.tf1 == "A") & (ov.table.tf2 == "D") & (ov.table.sign1 == "+") & (ov.table.sign2 == "-")]["overlap"].iloc[0]
    assert mat.loc["D", "A"] == ov.table[(ov.table.tf1 == "A") & (ov.table.tf2 == "D") & (ov.table.sign1 == "-") & (ov.table.sign2 == "+")]["overlap"].iloc[0]
    assert np.isnan(mat.loc["A", "A"])
    sym = ov.matrix("jaccard", ("+", "+"))
    np.testing.assert_allclose(sym.to_numpy(), sym.to_numpy().T, equal_nan=True)
    with pytest.raises(ValueError, match="Unknown metric"):
        ov.matrix("nope")
    with pytest.raises(ValueError, match="not in the table"):
        pr.regulon_overlap(reg, ["A", "B"]).matrix("jaccard", ("-", "-"))
    with pytest.raises(ValueError, match="Unknown regulator"):
        ov.shared("A", "Z")


def test_input_validation_and_universe(rng):
    reg = _regulon(rng)
    with pytest.raises(ValueError, match="tail"):
        pr.regulon_overlap(reg, ["A", "B"], tail="nope")
    with pytest.raises(ValueError, match="at least two"):
        pr.regulon_overlap(reg, ["A"])
    with pytest.raises(ValueError, match="unique"):
        pr.regulon_overlap(reg, ["A", "A"])
    with pytest.raises(ValueError, match="not in the regulon"):
        pr.regulon_overlap(reg, ["A", "ZZZ"])
    with pytest.raises(ValueError, match="duplicated"):
        pr.regulon_overlap(pd.concat([reg, reg.iloc[:3]]), ["A", "B"])
    with pytest.raises(ValueError, match="columns"):
        pr.regulon_overlap(reg.drop(columns="mor"), ["A", "B"])
    no_lik = pr.regulon_overlap(reg.drop(columns="likelihood"), ["A", "B"])
    assert (no_lik.likelihoods.stack() == 1.0).all()
    genes = sorted(reg["target"].unique())[:800]                                              # a smaller universe (e.g. expressed genes)
    small = pr.regulon_overlap(reg, ["A", "B"], universe=genes)
    assert small.universe_size == 800 and set(small.targets.index) <= set(genes)
    a = pr.regulon_overlap(reg, ["A", "B", "C"], n_perm=50, seed=3)
    b = pr.regulon_overlap(reg, ["A", "B", "C"], n_perm=50, seed=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert "RegulonOverlap(" in repr(a) and (a.table["q_hyper"] >= a.table["p_hyper"] - 1e-12).all()


def test_figures_and_statistics(rng):
    reg = _regulon(rng, discordant=25)
    ov = pr.regulon_overlap(reg, ["A", "B", "C", "D"], tail="both", n_perm=100)
    ax = ov.plot_heatmap("jaccard", ("+", "+"))
    assert len(ax.texts) == 0 and [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C", "D"]
    assert ov.add_heatmap_values(ax=ax, fmt="{:.3f}") is ax and len(ax.texts) == 12             # off-diagonal cells
    with pytest.raises(ValueError, match="plot_heatmap"):
        ov.add_heatmap_values(ax=plt.figure().add_subplot())
    sc = ov.plot_scatter("A", "B")
    assert len(sc.texts) == 0 and "likelihood" in sc.get_xlabel()
    assert "-/-" in ov.add_scatter_stats("A", "B", ax=sc).texts[0].get_text() and "p(perm)" in sc.texts[0].get_text()
    swapped = ov.add_scatter_stats("B", "A", ax=ov.plot_scatter("B", "A"))                      # reversed pair: roles of the signs swap
    assert "sign agreement" in swapped.texts[0].get_text()
    assert ov.plot_scatter("A", "D", value="mor", label_top=3) is not None
    with pytest.raises(ValueError, match="value must"):
        ov.plot_scatter("A", "B", value="nope")
    up = ov.plot_upset()
    assert len(up.texts) == 8 and ov.add_upset_counts(ax=up) is up and len(up.texts) > 8      # 8 sets (4 regulators x 2 tails)
    sel = ov.plot_upset(sets=["A+", "B+"], max_bars=3)
    assert [t.get_text().split()[0] for t in sel.texts] == ["A+", "B+"]
    with pytest.raises(ValueError, match="Unknown set"):
        ov.plot_upset(sets=["A"])
    with pytest.raises(ValueError, match="plot_upset"):
        ov.add_upset_counts(ax=plt.figure().add_subplot())
    pos = pr.regulon_overlap(reg, ["A", "B", "C"])
    assert [t.get_text().split()[0] for t in pos.plot_upset().texts] == ["A", "B", "C"]        # positive tail: one set per regulator
    _, axes = plt.subplots(1, 3)
    assert ov.plot_heatmap(ax=axes[0]) is axes[0] and ov.plot_scatter("A", "B", ax=axes[1]) is axes[1] and ov.plot_upset(ax=axes[2]) is axes[2]

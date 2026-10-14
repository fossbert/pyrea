"""Figures of PhenotypeComparison."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pyrea as pr
from pyrea.phenotypes_plot import group_colors


@pytest.fixture
def setup(rng):
    n = 150
    s = [f"s{i}" for i in range(n)]
    g = pd.Series(0.0, index=s)
    A, B, C = g.copy(), g.copy(), g.copy()
    A.iloc[:15] = 1
    B.iloc[15:30] = 1
    C.iloc[30:45] = 1
    B.iloc[10:12] = 1                 # overlap A / B
    C.iloc[145:] = np.nan
    act = pd.DataFrame(rng.normal(size=(2, n)), index=["TGT", "OTHER"], columns=s)
    rpt = pd.DataFrame(rng.normal(size=(2, n)), index=["TGT", "OTHER"], columns=s)
    act.loc["TGT", A[A == 1].index] -= 3
    act.loc["TGT", B[B == 1].index] -= 3
    return {"A": A, "B": B, "C": C}, act, rpt


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_result_carries_settings(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, method="cdf", n_boot=20, alpha=0.1, seed=1)
    assert (cmp.target, cmp.method, cmp.reference, cmp.groups) == ("TGT", "cdf", "neither", ("A", "B", "C"))
    assert cmp.alpha == 0.1 and cmp.n_boot == 20 and cmp.lr == 3.0
    cols = group_colors(cmp)
    assert set(cols) == {"neither", "A", "B", "C"} and len(set(cols.values())) == 4


def test_overview_figure_panels(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=50)
    fig = cmp.plot()
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titles) == 4                                   # A-D (+ the colour bar has no title)
    assert "TGT" in fig._suptitle.get_text() and "density" in fig._suptitle.get_text()
    one = pr.compare_phenotypes({"A": groups["A"]}, "TGT", act, rpt, n_boot=20).plot()
    assert len([ax for ax in one.axes if ax.get_title(loc="left")]) == 2
    assert pr.plot_phenocopy(cmp) is not None                 # function form


def test_cross_matrix_is_clean_until_stats_are_added(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=50)
    ax = cmp.plot_cross()
    assert len(ax.texts) == 0 and len(ax.patches) == 3                           # colours and dashed in-sample frames only
    cmp.add_cross_stats(ax=ax)
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 9 and all("\n[" in t for t in texts)               # AUC, interval, share
    assert all("mutant" not in t for t in [t.get_text() for t in cmp.add_cross_stats(ax=cmp.plot_cross(), share=False).texts])
    with pytest.warns(UserWarning):
        small = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=20, min_group=14)
    ax = small.add_cross_stats(ax=small.plot_cross())
    tx = [t.get_text() for t in ax.texts]
    assert tx.count("n/a") == 3 and sum("mutant" in t for t in tx) == 6             # A has 13 samples: no scale, "n/a" row
    off = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=0)
    assert all("[" not in t.get_text() for t in off.add_cross_stats(ax=off.plot_cross()).texts)   # no interval without bootstrap


def test_forest_rows_and_pairwise(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=50)
    ax = cmp.plot_auc()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 3 * (2 + 3) and "A | MPS[A]" in labels                # 3 groups x (G, RPT, 3 MPS scales)
    assert len(cmp.plot_auc(pairwise=True).get_yticklabels()) == 3 * 5 + 3 * 5
    only_g = cmp.plot_auc(scores=["G"])
    assert [t.get_text() for t in only_g.get_yticklabels()] == ["A | G", "B | G", "C | G"]
    assert pr.compare_phenotypes(groups, "TGT", act, n_boot=0).plot_auc() is not None   # no RPT, no intervals


def test_scores_and_scatter(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=20)
    ax = cmp.plot_scores("RPT")
    assert ax.get_ylabel() == "RPT" and len(ax.get_xticklabels()) == 4
    assert cmp.plot_scores("MPS[A]") is not None
    with pytest.raises(ValueError, match="Unknown score"):
        cmp.plot_scores("nope")
    ax = cmp.plot_scale_scatter("A", "C")
    assert "scale of A" in ax.get_xlabel() and "scale of C" in ax.get_ylabel()
    with pytest.raises(ValueError, match="Unknown group"):
        cmp.plot_scale_scatter("A", "Z")
    single = pr.compare_phenotypes({"A": groups["A"]}, "TGT", act, rpt, n_boot=0)
    with pytest.raises(ValueError, match="two groups"):
        single.plot_scale_scatter()


def test_scale_xy_for_custom_plots(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=0)
    xy, grp = cmp.scale_xy()
    assert list(xy.columns) == ["scale_A", "scale_B"] and xy.index.equals(grp.index)
    assert set(grp) <= {"neither", "A", "B", "C"} and "overlap" not in set(grp)      # overlap and unprofiled dropped
    assert xy.notna().all().all()
    xy2, _ = cmp.scale_xy("C", "A")
    np.testing.assert_allclose(xy2["scale_A"], cmp.scores.loc[xy2.index, "MPS[C]"])
    with pytest.raises(ValueError):
        pr.compare_phenotypes({"A": groups["A"]}, "TGT", act, rpt, n_boot=0).scale_xy()


def test_axes_are_accepted_and_returned(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=30)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    assert cmp.plot_scores("G", ax=axs[0]) is axs[0]
    assert cmp.plot_scale_scatter("A", "B", ax=axs[1]) is axs[1]
    assert cmp.plot_cross(ax=axs[2]) is axs[2]
    assert cmp.plot_auc(ax=axs[3]) is axs[3]
    axs[0].set_ylim(-20, 20)                                  # the caller can change the axes afterwards
    assert axs[0].get_ylim() == (-20, 20)
    fig2, axes = plt.subplots(2, 2)
    out = cmp.plot(axes=axes.ravel())
    assert [o is a for o, a in zip(out, axes.ravel())] == [True] * 4
    with pytest.raises(ValueError, match="Need 4 axes"):
        cmp.plot(axes=axes.ravel()[:3])


def test_panels_are_clean_and_statistics_are_added_afterwards(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=30)
    texts = lambda ax: [t.get_text() for t in ax.texts]
    ax = cmp.plot_scores("G")
    assert texts(ax) == []
    assert cmp.add_group_stats("G", ax=ax) is ax
    assert len([t for t in texts(ax) if t.startswith("AUC")]) == 3 and all(t.count("\n") == 2 for t in texts(ax))
    no_ci = cmp.add_group_stats("G", ax=cmp.plot_scores("G"), ci=False)
    assert all(t.get_text().count("\n") == 1 for t in no_ci.texts)           # AUC and p only
    sc = cmp.plot_scale_scatter("A", "B")
    assert texts(sc) == [] and any(t.startswith("AUC on x") for t in texts(cmp.add_scatter_stats(ax=sc)))
    sc2 = cmp.add_scatter_stats(ax=cmp.plot_scale_scatter("A", "C"), loc="lower left")
    assert "A " in texts(sc2)[0] and "C " in texts(sc2)[0].split("\n")[1]       # scales taken from the scatter
    ax = cmp.plot_auc()
    assert len(ax.figure.axes) == 1                                           # no twin axis yet
    cmp.add_auc_p(ax=ax)
    twin = [a for a in ax.figure.axes if a is not ax][0]
    assert len(twin.get_yticklabels()) == len(ax.get_yticklabels()) and all(t.get_text().startswith("p") for t in twin.get_yticklabels())
    with pytest.raises(ValueError, match="plot_auc"):
        cmp.add_auc_p(ax=cmp.plot_scores("G"))
    plt.close("all")
    clean = cmp.plot()
    assert all(len(ax.texts) == 0 or all(len(t.get_text()) == 1 for t in ax.texts) for ax in clean.axes)   # only the panel letters
    full = cmp.plot(stats=True)
    assert sum(len(ax.texts) for ax in full.axes) > sum(len(ax.texts) for ax in clean.axes) + 10


def test_pair_brackets_and_colors(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=30)
    ax = cmp.plot_scores("G")
    cmp.add_pair_stats("A", "B", "G", ax=ax)
    cmp.add_pair_stats("C", "A", "G", ax=ax, yoffset=0.2, auc=False, exact_p=True, line_kwargs={"color": "r"})   # either order
    texts = [t.get_text() for t in ax.texts]
    assert texts[0].startswith("AUC") and texts[1].startswith("p=") and "e" in texts[1]
    assert len(ax.lines) >= 2 and ax.get_ylim()[1] > max(l.get_ydata().max() for l in ax.lines[-2:]) - 1e-9    # headroom kept
    with pytest.raises(ValueError, match="Unknown group"):
        cmp.add_pair_stats("A", "Z", "G", ax=ax)
    one = cmp.add_group_stats("G", ax=cmp.plot_scores("G"), color="k", fontsize=8)          # text kwargs go to matplotlib
    assert all(t.get_color() == "k" and t.get_fontsize() == 8 for t in one.texts)
    cols = group_colors(cmp, {"A": "#123456"})
    assert cols["A"] == "#123456" and cols["B"] != cols["A"]
    sc = cmp.plot_scale_scatter("A", "B", colors={"A": "#123456"})
    face = [tuple(np.round(c.get_facecolor()[0][:3], 3)) for c in sc.collections if len(c.get_offsets())]
    assert (round(0x12 / 255, 3), round(0x34 / 255, 3), round(0x56 / 255, 3)) in face


def test_selecting_scores_and_groups(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=30)
    assert cmp.score_names == ["G", "RPT", "MPS[A]", "MPS[B]", "MPS[C]"]
    labels = lambda ax: [t.get_text() for t in ax.get_yticklabels()]
    assert labels(cmp.plot_auc(scores=["MPS[B]", "G"])) == ["A | MPS[B]", "B | MPS[B]", "C | MPS[B]", "A | G", "B | G", "C | G"]
    assert labels(cmp.plot_auc(scores=["G"], groups=["C", "A"])) == ["C | G", "A | G"]
    assert labels(cmp.plot_auc(scores=["G"], groups=["A", "B"], pairwise=True)) == ["A | G", "B | G", "A vs B | G"]
    with pytest.raises(ValueError, match="Unknown score"):
        cmp.plot_auc(scores=["nope"])
    with pytest.raises(ValueError, match="Unknown group"):
        cmp.plot_auc(groups=["A", "Z"])

    ax = cmp.plot_scores("G", groups=["B", "A"], reference=False)
    assert [t.get_text().split("\n")[0] for t in ax.get_xticklabels()] == ["B", "A"]
    cmp.add_group_stats("G", ax=ax)                                           # follows the positions shown
    assert len(ax.texts) == 2
    cmp.add_pair_stats("A", "B", "G", ax=ax)
    with pytest.raises(ValueError, match="must both be shown"):
        cmp.add_pair_stats("A", "C", "G", ax=ax)

    cx = cmp.plot_cross(groups=["A", "C"])
    assert len(cx.patches) == 2 and [t.get_text().split("\n")[0] for t in cx.get_xticklabels()] == ["A", "C"]
    assert len(cmp.add_cross_stats(ax=cx).texts) == 4
    sc = cmp.plot_scale_scatter("A", "B", groups=["C"])
    assert [h.get_text().split(" (")[0] for h in sc.get_legend().get_texts()] == ["neither", "C"]


def test_overview_with_selected_scores_and_groups(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=30)
    fig = cmp.plot(scores=["G", "MPS[A]"], groups=["A", "C"], stats=True)
    forest = [ax for ax in fig.axes if ax.get_title(loc="left").startswith("AUC with")][0]
    assert [t.get_text() for t in forest.get_yticklabels()] == ["A | G", "C | G", "A | MPS[A]", "C | MPS[A]",
                                                                 "A vs C | G", "A vs C | MPS[A]"]
    assert "A, C" in fig._suptitle.get_text()
    one = cmp.plot(groups=["B"], scores=["RPT"])                                # a single group: two panels
    assert len([ax for ax in one.axes if ax.get_title(loc="left")]) == 2

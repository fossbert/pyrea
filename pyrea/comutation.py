"""Does a second alteration (co-mutation) modify the effect of a driver on the activity of a protein?

Design: two binary alterations, the driver D and the partner C, give four cells of samples (neither, D only,
C only, D+C). For a target protein the activity ``y`` (G, optionally RPT and expression) is compared across the
cells. With the cell means ``mu_dc`` the simple effects are ``mu_11 - mu_10`` (C within D) and ``mu_01 - mu_00``
(C without D); their difference is the *interaction*, i.e. whether C changes what D does. A C effect that is the same
with and without D is a main effect of C, not a co-mutation effect.

Two views are reported side by side:

- **Rank-based effects** (``effects``): Cliff's delta (``2 * AUC - 1``) for the simple effects, with bootstrap
  intervals in which the four cells are resampled separately; the interaction on the delta scale is the difference of
  two deltas (descriptive: it depends on the scale).
- **Linear model** (``model``): ``y = a + b_D D + b_C C + b_DC D C + g'Z + e`` with covariates ``Z`` (e.g. subtype,
  cohort), heteroscedasticity-robust (HC3) standard errors and Freedman-Lane permutation p-values, which are valid
  for small cells. ``b_DC`` is the interaction on the additive (activity) scale.

Cell sizes are reported together with the smallest interaction that could be detected (``min_detectable_interaction``,
in residual SDs): a non-significant interaction with small cells says "not testable", not "absent".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, t as t_dist

from .concordance import _boot_cliff, _cliff

__all__ = ["comutation_effect", "CoMutationResult"]


@dataclass
class CoMutationResult:
    """Result of :func:`comutation_effect`.

    Attributes
    ----------
    scores : pd.DataFrame
        One row per sample used: ``cell`` (ordered categorical: neither, D only, C only, D+C), the scores of the
        target (``G``, ``RPT``, ``expr``) and the covariates.
    cells : pd.DataFrame
        Index = cell label: ``n``, and ``mean_<score>`` / ``median_<score>``.
    effects : pd.DataFrame
        Index (score, contrast): ``delta`` (Cliff's delta, > 0 = higher in the first group of the contrast) with the
        bootstrap interval ``low`` / ``high``, ``n1`` / ``n2``, and the Mann-Whitney ``p`` of the raw values. Contrasts:
        ``C within D`` (D+C vs. D only), ``C within not-D`` (C only vs. neither), ``D within C`` (D+C vs. C only),
        ``D within not-C`` (D only vs. neither), and the interactions ``interaction C by D`` (= delta of ``C within D``
        minus delta of ``C within not-D``) and ``interaction D by C``. With covariates the deltas are computed on values
        adjusted for them (see ``adjusted``).
    model : pd.DataFrame
        Index (score, term) with the terms D, C and ``D:C``: ``beta``, ``se`` (HC3), interval ``low`` / ``high``,
        ``p_hc3`` and ``p_perm`` (Freedman-Lane permutation test).
    target, driver, partner, cell_names, covariates :
        The settings of the analysis (cell labels in the order neither, D only, C only, D+C).
    n_boot, n_perm, alpha, min_cell :
        Bootstrap resamples, permutations, interval level, smallest cell for effects involving it.
    min_detectable_interaction : float
        Smallest interaction (in residual SDs) detectable with 80 % power at ``alpha`` given the four cell sizes.
    adjusted : bool
        Whether the rank-based effects were computed on covariate-adjusted values.
    n_dropped : int
        Samples dropped because a covariate was missing.

    Methods
    -------
    ``plot()`` (overview), ``plot_cells()``, ``plot_interaction()``, ``plot_effects()``: figures on an Axes that can be
    passed; ``add_cell_stats()``, ``add_interaction_stats()``, ``add_effect_p()``: statistics added afterwards; see
    :mod:`pyrea.comutation_plot`.
    """

    scores: pd.DataFrame
    cells: pd.DataFrame
    effects: pd.DataFrame
    model: pd.DataFrame
    target: str
    driver: str
    partner: str
    cell_names: tuple
    covariates: tuple = ()
    n_boot: int = 0
    n_perm: int = 0
    alpha: float = 0.05
    min_cell: int = 3
    min_detectable_interaction: float = float("nan")
    adjusted: bool = True
    n_dropped: int = 0

    def __repr__(self) -> str:
        n = ", ".join(f"{c}: {int(self.cells.loc[c, 'n'])}" for c in self.cell_names)
        return (f"CoMutationResult(target={self.target!r}, driver={self.driver!r}, partner={self.partner!r}; {n}; "
                f"min detectable interaction {self.min_detectable_interaction:.2f} SD)")

    # figures: see pyrea.comutation_plot
    def plot(self, score: str = "G", figsize=None, axes=None, stats: bool = False, colors: Optional[dict] = None,
             scores: Optional[Sequence[str]] = None):
        """Three-panel overview: ``plot_cells``, ``plot_interaction``, ``plot_effects``.

        Returns the figure, or the list of the three axes if ``axes`` is given. ``stats=True`` adds all statistics."""
        from .comutation_plot import plot_comutation
        return plot_comutation(self, score, figsize, axes, stats, colors, scores)

    def plot_cells(self, score: str = "G", ax=None, seed: int = 0, title: Optional[str] = None,
                   colors: Optional[dict] = None):
        """Score per cell (neither, D only, C only, D+C), one point per sample."""
        from .comutation_plot import plot_cells
        return plot_cells(self, score, ax, seed, title, colors)

    def plot_interaction(self, score: str = "G", ax=None, title: Optional[str] = None, colors: Optional[dict] = None):
        """Interaction plot: cell means (95 % interval) for C absent and present across D absent / present."""
        from .comutation_plot import plot_interaction
        return plot_interaction(self, score, ax, title, colors)

    def plot_effects(self, ax=None, scores: Optional[Sequence[str]] = None, contrasts: Optional[Sequence[str]] = None,
                     title: Optional[str] = None):
        """Forest plot of the rank-based effects (Cliff's delta with bootstrap interval)."""
        from .comutation_plot import plot_effects
        return plot_effects(self, ax, scores, contrasts, title)

    def add_cell_stats(self, group_a: str, group_b: str, score: str = "G", ax=None, yoffset: float = 0.08,
                       exact_p: bool = False, line_kwargs: Optional[dict] = None, **text_kwargs):
        """After ``plot_cells``: bracket between two cells with the delta and Mann-Whitney p of their comparison."""
        from .comutation_plot import add_cell_stats
        return add_cell_stats(self, group_a, group_b, score, ax, yoffset, exact_p, line_kwargs, **text_kwargs)

    def add_interaction_stats(self, score: str = "G", ax=None, loc: str = "lower right", **text_kwargs):
        """After ``plot_interaction``: the interaction coefficient with its interval and permutation p as a text box."""
        from .comutation_plot import add_interaction_stats
        return add_interaction_stats(self, score, ax, loc, **text_kwargs)

    def add_effect_p(self, ax=None, exact_p: bool = False, **text_kwargs):
        """After ``plot_effects``: the Mann-Whitney p of every row at the right edge."""
        from .comutation_plot import add_effect_p
        return add_effect_p(self, ax, exact_p, **text_kwargs)


# --------------------------------------------------------------------------------------------- helpers
def _covariate_matrix(cov, index: pd.Index) -> pd.DataFrame:
    """Covariates as a numeric design block: categorical columns dummy-coded (first level dropped), constants removed."""
    if cov is None:
        return pd.DataFrame(index=index)
    cov = cov.to_frame() if isinstance(cov, pd.Series) else pd.DataFrame(cov)
    cov = cov.reindex(index)
    parts = []
    for c in cov.columns:
        col = cov[c]
        if pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype):
            parts.append(col.astype(float).rename(str(c)))
        else:
            parts.append(pd.get_dummies(col.astype(object), prefix=str(c), prefix_sep="=", drop_first=True, dtype=float))
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)
    return out.loc[:, out.nunique() > 1]


def _ols(X: np.ndarray, y: np.ndarray):
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    return beta, resid, h, xtx_inv


def _drop_collinear(Z: pd.DataFrame, base: np.ndarray):
    """Drop covariate columns that add no rank to the design (``base`` plus the columns kept so far); returns ``(Z, dropped names)``."""
    keep, dropped = [], []
    cur = base
    rank = np.linalg.matrix_rank(cur)
    for col in Z.columns:
        trial = np.column_stack([cur, Z[col].to_numpy(dtype=float)])
        r = np.linalg.matrix_rank(trial)
        if r > rank:
            keep.append(col)
            cur, rank = trial, r
        else:
            dropped.append(str(col))
    return Z[keep], dropped


def _hc3_se(X: np.ndarray, resid: np.ndarray, h: np.ndarray, xtx_inv: np.ndarray) -> np.ndarray:
    # leverage 1 (a covariate level with a single sample): that sample is fitted exactly, its residual is ~0
    meat = X.T @ (X * (resid / np.clip(1 - h, 1e-6, None))[:, None] ** 2)
    return np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))


def _freedman_lane_p(X: np.ndarray, y: np.ndarray, k: int, n_perm: int, rng) -> float:
    """Permutation p-value of coefficient ``k`` (Freedman-Lane): permute the residuals of the model without it."""
    n, p = X.shape
    beta, resid, _, xtx_inv = _ols(X, y)
    s2 = resid @ resid / (n - p)
    t_obs = beta[k] / np.sqrt(s2 * xtx_inv[k, k])
    Xr = np.delete(X, k, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, res_r = Xr @ br, y - Xr @ br
    count = 0
    block = max(1, min(n_perm, 2_000_000 // n))
    done = 0
    proj = xtx_inv @ X.T
    while done < n_perm:
        b = min(block, n_perm - done)
        idx = rng.random((n, b)).argsort(axis=0)
        ystar = fit_r[:, None] + res_r[idx]
        bstar = proj @ ystar
        rss = ((ystar - X @ bstar) ** 2).sum(axis=0)
        tstar = bstar[k] / np.sqrt(rss / (n - p) * xtx_inv[k, k])
        count += int(np.sum(np.abs(tstar) >= abs(t_obs) - 1e-12))
        done += b
    return (count + 1) / (n_perm + 1)


def _percentile_ci(v: np.ndarray, alpha: float):
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def comutation_effect(driver: pd.Series, partner: pd.Series, target: str, activity: pd.DataFrame,
                      rpt: Optional[pd.DataFrame] = None, expression: Optional[pd.DataFrame] = None,
                      covariates=None, names: Optional[Sequence[str]] = None, min_cell: int = 3,
                      n_boot: int = 1000, n_perm: int = 2000, alpha: float = 0.05, adjusted: bool = True,
                      seed: int = 0) -> CoMutationResult:
    """Effect of a co-alteration on the activity of a protein, alone and in the presence of a driver.

    Parameters
    ----------
    driver, partner : pd.Series
        Alterations per sample: 1 = present, 0 = absent, NaN = not profiled (excluded). Use e.g. rows of
        ``cbk.alteration_matrix`` (a patient with a deletion of the gene should be NaN, not 0, in a mutation row).
    target : str
        Protein whose G-activity (``activity``), RPT-activity (``rpt``) and expression (``expression``) are
        compared; a row in each matrix (genes x samples).
    covariates : pd.Series or pd.DataFrame, optional
        Per sample, e.g. molecular subtype and cohort. Categorical columns are dummy-coded. They enter the linear model;
        the rank-based effects are computed on values adjusted for them (``adjusted=True``: residuals of ``y`` on the
        covariates plus the overall mean). Samples with a missing covariate are dropped (``n_dropped``). With small cells
        use additive adjustment like this, not full stratification.
    names : sequence of 2 str, optional
        Labels of driver and partner (default: the Series names, else ``"driver"`` / ``"partner"``).
    min_cell : int
        Effects involving a cell with fewer samples are NaN (a warning names the cell).
    n_boot, n_perm : int
        Bootstrap resamples for the rank-based intervals (the four cells are resampled separately; the same resample for
        all scores and contrasts) and permutations for the Freedman-Lane p-values (0 switches them off).
    alpha : float
        Level of the intervals (``1 - alpha`` percentile / t intervals).
    seed : int

    Returns
    -------
    CoMutationResult
        See there for the tables. The interaction is the quantity of interest; the main effect of the partner
        (``C within not-D``) tells whether the partner changes the activity regardless of the driver.

    Notes
    -----
    Cells of a few samples give wide intervals and low power; ``min_detectable_interaction`` reports what the design can
    detect. Many partners: run this per partner and correct for multiple testing yourself (BH).

    Examples
    --------
    ::

        mat = cbk.alteration_matrix(ex, types=("MUT",), by="PATIENT_ID")
        res = pr.comutation_effect(mat.loc["RHOA"], mat.loc["ARID1A"], "RHOA", vpres, rpt=rpt, expression=emat,
                                   covariates=meta[["SUBTYPE"]])
        res.cells; res.effects; res.model
        res.plot(stats=True)
    """
    rng = np.random.default_rng(seed)
    dname = (str(driver.name) if names is None and driver.name is not None else (names[0] if names else "driver"))
    pname = (str(partner.name) if names is None and partner.name is not None else (names[1] if names else "partner"))
    if dname == pname:
        raise ValueError("driver and partner need different names")
    if target not in activity.index:
        raise ValueError(f"Target {target!r} has no activity (not in the regulon)")
    traits = {"G": activity}
    if rpt is not None and target in rpt.index:
        traits["RPT"] = rpt
    if expression is not None and target in expression.index:
        traits["expr"] = expression

    idx = driver.index.intersection(partner.index, sort=False)
    for m in traits.values():
        idx = idx.intersection(m.columns, sort=False)
    d = driver.reindex(idx).astype(float)
    c = partner.reindex(idx).astype(float)
    y = pd.DataFrame({k: m.loc[target, idx].astype(float) for k, m in traits.items()})
    ok = d.notna() & c.notna() & y.notna().all(axis=1)
    Z = _covariate_matrix(covariates, idx)
    n_dropped = 0
    if covariates is not None:
        cov_ok = pd.DataFrame(covariates).reindex(idx).notna().all(axis=1) if not isinstance(covariates, pd.Series) \
            else covariates.reindex(idx).notna()
        n_dropped = int((ok & ~cov_ok).sum())
        ok &= cov_ok
    d, c, y, Z = d[ok].gt(0).astype(float), c[ok].gt(0).astype(float), y[ok], Z[ok]
    Z = Z.loc[:, Z.nunique() > 1]
    Z, dropped = _drop_collinear(Z, np.column_stack([np.ones(len(d)), d.to_numpy(), c.to_numpy(), (d * c).to_numpy()]))
    if dropped:
        warnings.warn("Covariate columns dropped because they are linear combinations of others (e.g. subtypes nested in "
                      "cohorts): %s" % ", ".join(dropped), UserWarning, stacklevel=2)
    n = len(d)
    if n < 8:
        raise ValueError(f"Only {n} samples with all values; cannot fit the design")

    labels = ["neither", f"{dname} only", f"{pname} only", f"{dname}+{pname}"]
    cell = np.select([(d == 1) & (c == 1), (d == 1) & (c == 0), (d == 0) & (c == 1)],
                     [labels[3], labels[1], labels[2]], default=labels[0])
    cell = pd.Categorical(cell, categories=labels, ordered=True)
    counts = pd.Series(cell).value_counts().reindex(labels)
    small = [f"{k!r} (n={int(v)})" for k, v in counts.items() if v < min_cell]
    if small:
        warnings.warn("Cells with fewer than min_cell=%d samples; effects involving them are NaN: %s"
                      % (min_cell, ", ".join(small)), UserWarning, stacklevel=2)

    scores = y.copy()
    scores.insert(0, "cell", pd.Series(cell, index=y.index))
    for col in (pd.DataFrame(covariates).reindex(y.index).columns if covariates is not None else []):
        scores[str(col)] = pd.DataFrame(covariates).reindex(y.index)[col]

    # ---- cell table
    cells = pd.DataFrame({"n": counts.astype(int)})
    for s in y.columns:
        grp = y[s].groupby(pd.Series(cell, index=y.index), observed=False)
        cells[f"mean_{s}"] = grp.mean().reindex(labels)
        cells[f"median_{s}"] = grp.median().reindex(labels)

    # ---- values for the rank-based effects (adjusted for covariates if given)
    yv = y.copy()
    if adjusted and Z.shape[1]:
        A = np.column_stack([np.ones(n), Z.to_numpy()])
        for s in y.columns:
            coef = np.linalg.lstsq(A, y[s].to_numpy(), rcond=None)[0]
            yv[s] = y[s].to_numpy() - A @ coef + y[s].mean()

    ci_ = {k: np.flatnonzero(np.asarray(cell == k)) for k in labels}
    cnames = {"C within D": (labels[3], labels[1]), "C within not-D": (labels[2], labels[0]),
              "D within C": (labels[3], labels[2]), "D within not-C": (labels[1], labels[0])}
    boots = {k: rng.integers(0, max(len(v), 1), (n_boot, len(v))) for k, v in ci_.items()} if n_boot else None
    rows = {}
    for s in y.columns:
        vals = {k: yv[s].to_numpy()[v] for k, v in ci_.items()}
        point, boot = {}, {}
        for name, (a, b) in cnames.items():
            if len(vals[a]) >= min_cell and len(vals[b]) >= min_cell:
                point[name] = _cliff(vals[a], vals[b])
                boot[name] = _boot_cliff(vals[a], vals[b], boots[a], boots[b]) if n_boot else None
                p = float(mannwhitneyu(y[s].to_numpy()[ci_[a]], y[s].to_numpy()[ci_[b]]).pvalue)
            else:
                point[name], boot[name], p = np.nan, None, np.nan
            lo, hi = _percentile_ci(boot[name], alpha) if boot[name] is not None else (np.nan, np.nan)
            rows[(s, name)] = {"delta": point[name], "low": lo, "high": hi, "n1": len(vals[a]), "n2": len(vals[b]), "p": p}
        for name, (p1, p2) in {"interaction C by D": ("C within D", "C within not-D"),
                               "interaction D by C": ("D within C", "D within not-C")}.items():
            est = point[p1] - point[p2]
            if boot[p1] is not None and boot[p2] is not None:
                lo, hi = _percentile_ci(boot[p1] - boot[p2], alpha)
            else:
                lo = hi = np.nan
            rows[(s, name)] = {"delta": est, "low": lo, "high": hi, "n1": np.nan, "n2": np.nan, "p": np.nan}
    effects = pd.DataFrame.from_dict(rows, orient="index")
    effects.index = pd.MultiIndex.from_tuples(effects.index, names=["score", "contrast"])

    # ---- linear model with interaction
    terms = [dname, pname, f"{dname}:{pname}"]
    X = np.column_stack([np.ones(n), d.to_numpy(), c.to_numpy(), (d * c).to_numpy()] + ([Z.to_numpy()] if Z.shape[1] else []))
    mrows = {}
    full_rank = np.linalg.matrix_rank(X) == X.shape[1] and n > X.shape[1]
    if not full_rank:
        warnings.warn("The design is rank deficient (an empty cell or collinear covariates); no model fitted",
                      UserWarning, stacklevel=2)
    tcrit = t_dist.ppf(1 - alpha / 2, max(n - X.shape[1], 1))
    for s in y.columns:
        ys = y[s].to_numpy()
        if full_rank:
            beta, resid, h, xtx_inv = _ols(X, ys)
            se = _hc3_se(X, resid, h, xtx_inv)
        for j, term in enumerate(terms, start=1):
            if not full_rank:
                mrows[(s, term)] = dict.fromkeys(["beta", "se", "low", "high", "p_hc3", "p_perm"], np.nan)
                continue
            tstat = beta[j] / se[j]
            mrows[(s, term)] = {"beta": beta[j], "se": se[j], "low": beta[j] - tcrit * se[j], "high": beta[j] + tcrit * se[j],
                                "p_hc3": float(2 * t_dist.sf(abs(tstat), n - X.shape[1])),
                                "p_perm": _freedman_lane_p(X, ys, j, n_perm, rng) if n_perm else np.nan}
    model = pd.DataFrame.from_dict(mrows, orient="index")
    model.index = pd.MultiIndex.from_tuples(model.index, names=["score", "term"])

    # ---- smallest detectable interaction (80 % power), in residual SDs
    mde = float((norm.ppf(1 - alpha / 2) + norm.ppf(0.8)) * np.sqrt(np.sum(1.0 / counts.clip(lower=1).to_numpy())))
    if (counts == 0).any():
        mde = float("nan")
    return CoMutationResult(scores=scores, cells=cells, effects=effects, model=model, target=target, driver=dname,
                            partner=pname, cell_names=tuple(labels),
                            covariates=tuple(str(c_) for c_ in (pd.DataFrame(covariates).columns if covariates is not None else [])),
                            n_boot=n_boot, n_perm=n_perm, alpha=alpha, min_cell=min_cell,
                            min_detectable_interaction=mde, adjusted=adjusted, n_dropped=n_dropped)

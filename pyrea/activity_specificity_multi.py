"""Activity specificity of regulator sets across many cohorts (e.g. one TCGA tumour type with its own network each).

:func:`activity_specificity_across_cohorts` infers the VIPER activity of the **whole network** in every cohort (needs R), evaluates
:func:`pyrea.activity_specificity` there (raw and adjusted for covariates such as the tumour purity), and adds an activity from a
regulon restricted to the candidates with VIPER's pleiotropy ("shadow") correction. The cohorts are combined in a
:class:`MultiActivity`: tables with the column ``cohort``, a summary, a random-effects meta-analysis of the correlations of the pairs, and
figures. Names, ``n_jobs``, cache and merging work as in :func:`pyrea.overlap_across_networks`.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .activity_specificity import _check_sets, activity_specificity
from .covariates import _flatten, load_covariates, resolve_covariates
from .regulon_overlap_multi import _run_all

__all__ = ["activity_specificity_across_cohorts", "MultiActivity", "tcga_sample_id"]

_SETTING_KEYS = ("method", "minsize", "direction", "pleiotropy", "min_samples", "score_minsize")
_SPEARMAN_VAR = 1.06        # variance inflation of Fisher's z for Spearman correlations (Fieller, Hartley, Pearson 1957)


def _meta(rho: np.ndarray, n: np.ndarray, q: int) -> dict:
    """DerSimonian-Laird random-effects meta-analysis of correlations on Fisher's z scale."""
    ok = np.isfinite(rho) & (n - 3 - q > 0)
    rho, n = rho[ok], n[ok]
    k = len(rho)
    if k == 0:
        return dict(k=0, rho_pooled=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, tau2=np.nan, I2=np.nan, p_heterogeneity=np.nan)
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    v = _SPEARMAN_VAR / (n - 3 - q)
    w = 1 / v
    zf = np.sum(w * z) / np.sum(w)
    Q = float(np.sum(w * (z - zf) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (v + tau2)
    zr = float(np.sum(ws * z) / np.sum(ws))
    se = float(np.sqrt(1 / np.sum(ws)))
    return dict(k=k, rho_pooled=float(np.tanh(zr)), ci_low=float(np.tanh(zr - 1.96 * se)), ci_high=float(np.tanh(zr + 1.96 * se)),
                p=float(2 * norm.sf(abs(zr / se))), tau2=float(tau2), I2=float(max(0.0, (Q - (k - 1)) / Q)) if k > 1 and Q > 0 else 0.0,
                p_heterogeneity=float(chi2.sf(Q, k - 1)) if k > 1 else np.nan)


def tcga_sample_id(name: str, length: int = 15) -> str:
    """TCGA sample name for matching: dots to dashes, cut to ``length`` characters (``"TCGA.2H.A9GF.01A.11R.A37I.31"`` ->
    ``"TCGA-2H-A9GF-01"``: patient and sample type). A module-level function, so it works as ``sample_id`` with ``n_jobs > 1``."""
    return str(name).replace(".", "-")[:length]


@dataclass
class MultiActivity:
    """Result of :func:`activity_specificity_across_cohorts`.

    Attributes
    ----------
    results : dict
        Cohort name -> :class:`pyrea.ActivitySpecificity`.
    skipped : dict
        Cohort name -> reason, for the cohorts that contain none of the requested sets.
    coverage : pd.DataFrame
        One row per cohort: the ``network`` and ``expression`` (paths), ``n_sets`` evaluated, ``missing`` regulators, ``n_samples`` and
        ``n_dropped`` (samples without covariates), and ``pleiotropy`` (``"ok"``, ``"failed: ..."`` or ``"off"``).
    pairs, table, extension : pd.DataFrame
        The tables of :class:`pyrea.ActivitySpecificity` of all cohorts, with ``cohort`` as first column.
    settings : dict
        The settings and the sets requested.

    Methods
    -------
    ``summary()``, ``meta()``, ``matrix()``, ``pair_matrix()``: tables. ``merge()``, ``concat()``, ``export()``. ``plot_matrix()``,
    ``add_top()``, ``plot_forest()``: figures.
    """

    results: dict
    skipped: dict
    coverage: pd.DataFrame
    pairs: pd.DataFrame
    table: pd.DataFrame
    extension: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"MultiActivity({len(self.results)} cohort(s), {len(self.skipped)} skipped, "
                f"{self.table['set'].nunique() if len(self.table) else 0} set(s), covariates={self.settings.get('covariates')})")

    # ------------------------------------------------------------------ combining
    @classmethod
    def concat(cls, parts: Sequence["MultiActivity"]) -> "MultiActivity":
        """Combine analyses of **different cohorts** (matched by name): same ``method``, ``minsize``, ``direction``, ``pleiotropy``,
        ``min_samples`` and covariates, and no repeated cohort name (an error otherwise); the sets may differ."""
        parts = list(parts)
        if not parts:
            raise ValueError("Nothing to concatenate")
        ref = parts[0].settings
        for i, m in enumerate(parts[1:], 1):
            diff = [k for k in _SETTING_KEYS + ("covariates",) if m.settings.get(k) != ref.get(k)]
            if diff:
                raise ValueError(f"Analysis {i} differs in {diff} from the first one; combine only analyses with the same settings")
        names = [n for m in parts for n in list(m.results) + list(m.skipped)]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"Cohort names occur in more than one analysis: {dup}; rename them")
        settings = dict(ref)
        settings["sets"] = list(dict.fromkeys(s for m in parts for s in m.settings.get("sets", [])))
        settings["n_cohorts"] = sum(m.settings.get("n_cohorts", len(m.results) + len(m.skipped)) for m in parts)
        cat = lambda a: pd.concat([getattr(m, a) for m in parts], ignore_index=True)
        return cls(results={n: r for m in parts for n, r in m.results.items()}, skipped={n: w for m in parts for n, w in m.skipped.items()},
                   coverage=cat("coverage"), pairs=cat("pairs"), table=cat("table"), extension=cat("extension"), settings=settings)

    def merge(self, *others: "MultiActivity") -> "MultiActivity":
        """This analysis combined with others of different cohorts (see :meth:`concat`)."""
        return MultiActivity.concat([self, *others])

    # ------------------------------------------------------------------ tables
    def _adj(self) -> bool:
        return "padj_ext_percentile_max" in self.table.columns

    def summary(self, top_frac: float = 0.05) -> pd.DataFrame:
        """Per set: in how many cohorts it was evaluated (``n_cohorts``), the median pairwise correlation (``median_rho``, and
        ``median_partial_rho`` with covariates), the median worst partner percentile of a member (``median_ext_percentile``; adjusted
        ``median_padj_ext_percentile`` with covariates), the cohorts in which every member is within the best ``top_frac`` of the partners of
        the others (``n_top``; adjusted ones if covariates were given) and in which every member is the best partner (``n_mutual_best``)."""
        t = self.table
        pre = "padj_" if self._adj() else ""
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key, d in t.groupby("set", sort=False):
                rec = {"set": key, "n_cohorts": len(d), "median_rho": float(d["rho_mean"].median()),
                       "median_ext_percentile": float(d["ext_percentile_max"].median())}
                if self._adj():
                    rec["median_partial_rho"] = float(d["partial_rho_mean"].median())
                    rec["median_padj_ext_percentile"] = float(d["padj_ext_percentile_max"].median())
                rec["n_top"] = int((d[pre + "ext_percentile_max"] <= top_frac).sum())
                rec["n_mutual_best"] = int(d[pre + "mutual_best"].sum())
                rec["cohorts_top"] = ";".join(d.loc[d[pre + "ext_percentile_max"] <= top_frac, "cohort"])
                rows.append(rec)
        return pd.DataFrame(rows).sort_values(["n_top", "median_rho"], ascending=False, ignore_index=True)

    def meta(self, variant: str = "full", adjusted: Optional[bool] = None) -> pd.DataFrame:
        """Random-effects meta-analysis (DerSimonian-Laird on Fisher's z, variance inflated by 1.06 for Spearman correlations) of the
        correlation of every pair over the cohorts: ``k`` cohorts, pooled ``rho_pooled`` with 95 % CI and ``p``, ``tau2``, ``I2`` and
        ``p_heterogeneity``, plus the ``rho_median`` and ``n_total`` samples. ``variant``: ``"full"`` (activity of the whole network) or
        ``"pleiotropy"``; ``adjusted``: ``False`` = Spearman, ``True`` = partial (covariates), ``None`` = both if available. The cohorts
        are different tumour types, so the pooled value is an average over cohorts; the heterogeneity says whether one number is justified."""
        pa = self.pairs[self.pairs["variant"] == variant]
        if pa.empty:
            raise ValueError(f"No pairs for variant {variant!r}; available: {sorted(self.pairs['variant'].unique())}")
        modes = [False] + ([True] if "partial_rho" in pa.columns else [])
        if adjusted is not None:
            if adjusted and "partial_rho" not in pa.columns:
                raise ValueError("The analysis has no covariates (no partial correlations)")
            modes = [bool(adjusted)]
        q = len(self.settings.get("covariates") or [])
        rows = []
        for (a, b), d in pa.groupby(["tf1", "tf2"], sort=False):
            for adj in modes:
                col = "partial_rho" if adj else "rho"
                rec = {"tf1": a, "tf2": b, "variant": variant, "adjusted": adj}
                rec.update(_meta(d[col].to_numpy(dtype=float), d["n"].to_numpy(dtype=float), q if adj else 0))
                rec["rho_median"] = float(d[col].median())
                rec["n_total"] = int(d["n"].sum())
                rows.append(rec)
        return pd.DataFrame(rows)

    def matrix(self, value: str = "padj_ext_percentile_max") -> pd.DataFrame:
        """Cohorts (rows) x sets (columns) of one column of ``table`` (default: the adjusted worst partner percentile if covariates were
        given, else ``ext_percentile_max``)."""
        value = value if value != "padj_ext_percentile_max" or self._adj() else "ext_percentile_max"
        if value not in self.table.columns:
            raise ValueError(f"Unknown value {value!r}; columns: {list(self.table.columns)}")
        mat = self.table.pivot(index="cohort", columns="set", values=value)
        return mat.reindex(index=[n for n in self.results if n in mat.index], columns=list(dict.fromkeys(self.table["set"])))

    def pair_matrix(self, value: str = "partial_rho", variant: str = "full") -> pd.DataFrame:
        """Cohorts (rows) x pairs (columns ``"TF1 | TF2"``) of one column of ``pairs`` (``rho``, ``partial_rho``, ``p``, ``partial_p``)
        for one activity variant."""
        pa = self.pairs[self.pairs["variant"] == variant]
        if value not in pa.columns:
            raise ValueError(f"Unknown value {value!r}; columns: {list(pa.columns)}")
        pa = pa.assign(pair=pa["tf1"] + " | " + pa["tf2"])
        mat = pa.pivot(index="cohort", columns="pair", values=value)
        return mat.reindex(index=[n for n in self.results if n in mat.index], columns=list(dict.fromkeys(pa["pair"])))

    def pleiotropy_effect(self) -> pd.DataFrame:
        """Per cohort and pair: Spearman ``rho`` of the full-network activity, with the pleiotropy correction (``rho_pleiotropy``, from a
        regulon restricted to the candidates) and the change; cohorts without a successful correction are left out."""
        pa = self.pairs
        if "pleiotropy" not in set(pa["variant"]):
            raise ValueError("The analysis has no pleiotropy variant (pleiotropy=False or it failed in every cohort)")
        keys = ["cohort", "tf1", "tf2"]
        full = pa[pa["variant"] == "full"].set_index(keys)
        pl = pa[pa["variant"] == "pleiotropy"].set_index(keys)
        out = pd.DataFrame({"rho_full": full["rho"], "rho_pleiotropy": pl["rho"]}).dropna()
        out["change"] = out["rho_pleiotropy"] - out["rho_full"]
        if "partial_rho" in pa.columns:
            out["partial_rho_full"], out["partial_rho_pleiotropy"] = full["partial_rho"], pl["partial_rho"]
        return out.reset_index()

    # ------------------------------------------------------------------ export and figures
    def export(self, path: Union[str, Path]) -> Path:
        """One Excel workbook (``.xlsx``) or a folder of CSV files: ``settings``, ``coverage``, ``summary``, ``meta`` (and ``meta_adjusted``
        with covariates), ``table``, ``pairs``, ``extension``."""
        path = Path(path)
        tables = {"settings": pd.DataFrame([{k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in self.settings.items()}]),
                  "coverage": self.coverage, "summary": self.summary(), "meta": self.meta(adjusted=False)}
        if self._adj():
            tables["meta_adjusted"] = self.meta(adjusted=True)
        tables.update(table=self.table, pairs=self.pairs, extension=self.extension)
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for name, df in tables.items():
                    df.to_excel(xw, sheet_name=name[:31], index=False)
        else:
            path.mkdir(parents=True, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(path / f"{name}.csv", index=False)
        return path

    def plot_matrix(self, value: str = "padj_ext_percentile_max", ax=None, cmap: Optional[str] = None, title: Optional[str] = None):
        """Cohorts x sets heatmap of one column of ``table``; mark the sets in the best partners with :meth:`add_top`."""
        from .activity_specificity_plot import plot_matrix
        return plot_matrix(self, value, ax, cmap, title)

    def add_top(self, ax=None, top_frac: float = 0.05, marker: str = "*", **text_kwargs):
        """After :meth:`plot_matrix`: mark the cells where every member is within the best ``top_frac`` of the partners of the others."""
        from .activity_specificity_plot import add_top
        return add_top(self, ax, top_frac, marker, **text_kwargs)

    def plot_forest(self, tf1: str, tf2: str, variant: str = "full", adjusted: bool = False, ax=None, title: Optional[str] = None):
        """Correlation of one pair per cohort (95 % interval on Fisher's z) with the pooled random-effects estimate at the bottom."""
        from .activity_specificity_plot import plot_forest
        return plot_forest(self, tf1, tf2, variant, adjusted, ax, title)


# ---------------------------------------------------------------------- running the cohorts
def _read_expression(item, reader):
    if isinstance(item, pd.DataFrame):
        return item
    if reader is not None:
        return reader(item)
    return pd.read_csv(item, sep="\t", index_col=0)


def _read_regulon(item, reader):
    if isinstance(item, pd.DataFrame):
        return item
    if reader is not None:
        return reader(item)
    from .rbackend.regulon import read_regulon_rds
    return read_regulon_rds(item)


def _spec_key(spec) -> str:
    """Digest of the covariate specification for the cache key (values of tables and registry variables, gene lists)."""
    h = hashlib.sha1()
    for p in _flatten(spec):
        if isinstance(p, str):
            h.update(("var:" + p).encode())
            try:
                h.update(pd.util.hash_pandas_object(load_covariates([p]), index=True).to_numpy().tobytes())
            except ValueError:
                pass
        elif isinstance(p, Mapping):
            h.update(repr(sorted((str(k), tuple(map(str, v))) for k, v in p.items())).encode())
        else:
            f = p.to_frame() if isinstance(p, pd.Series) else p
            h.update(repr(list(f.columns)).encode() + pd.util.hash_pandas_object(f, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def _stat_key(item) -> tuple:
    if isinstance(item, pd.DataFrame):
        return ("frame", hashlib.sha1(pd.util.hash_pandas_object(item, index=True).to_numpy().tobytes()).hexdigest()[:16])
    st = Path(item).stat()
    return (str(Path(item).resolve()), st.st_size, st.st_mtime_ns)


def _run_cohort(name, item, sets, kw, covariates, sample_id, cache_dir, expression_reader, regulon_reader):
    network, expression = item
    cov_key = None if covariates is None else _spec_key(covariates)
    key = hashlib.sha1(repr((name, tuple(sets), tuple(sorted(kw.items())), cov_key, _stat_key(network), _stat_key(expression),
                             getattr(sample_id, "__name__", None) if sample_id else None, "activity")).encode()).hexdigest()[:16]
    f = None
    if cache_dir is not None:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        f = Path(cache_dir) / f"{safe}-{key}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                return pickle.load(fh)

    from .rbackend.viper import viper_activity
    reg = _read_regulon(network, regulon_reader)
    expr = _read_expression(expression, expression_reader)
    if sample_id is not None:
        expr = expr.set_axis([sample_id(c) for c in expr.columns], axis=1)
    n_dup = int(expr.columns.duplicated().sum())
    if n_dup:
        expr = expr.loc[:, ~expr.columns.duplicated()]
    have = set(reg["source"].astype(str))
    present = [s for s in sets if all(t in have for t in s)]
    missing = sorted({t for s in sets for t in s} - have)
    info = {"missing": missing, "n_sets": 0, "n_samples": expr.shape[1], "n_dropped": 0, "pleiotropy": "off", "n_duplicated_samples": n_dup,
            "reason": ""}
    out = None
    if present:
        act = viper_activity(expr, reg, method=kw["method"], minsize=kw["minsize"], eset_filter=False)
        ok = [s for s in present if all(t in act.index for t in s)]
        lost = sorted({t for s in present for t in s if t not in act.index})
        info["missing"] = sorted(set(missing) | set(lost))
        if ok:
            members = sorted({t for s in ok for t in s})
            variants = {}
            if kw["pleiotropy"]:
                try:
                    restricted = reg[reg["source"].astype(str).isin(members)]
                    variants["pleiotropy"] = viper_activity(expr, restricted, method=kw["method"], minsize=kw["minsize"], eset_filter=False,
                                                            pleiotropy=True, pleiotropy_args=kw["pleiotropy_args"])
                    info["pleiotropy"] = "ok"
                except Exception as exc:                                   # viper's R code can fail for very few regulators
                    info["pleiotropy"] = "failed: " + str(exc).strip().splitlines()[0][:120]
                    variants = {}
            variants = {k: v.reindex(members) for k, v in variants.items()}
            if any(v.isna().any().any() for v in variants.values()):
                info["pleiotropy"] = "failed: regulators without activity"
                variants = {}
            cov = resolve_covariates(covariates, expr, method=kw["method"], min_genes=kw.get("score_minsize", 10))
            out = activity_specificity(act, ok, covariates=cov, variants=variants, direction=kw["direction"], top=kw["top"],
                                       min_samples=kw["min_samples"])
            info["n_sets"], info["n_samples"], info["n_dropped"] = len(ok), out.settings["n_samples"], out.settings["n_dropped"]
    if out is None:
        info["reason"] = "none of the requested sets is in the network"
    res = (name, out, info)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(res, fh)
    return res


def _say(run) -> None:
    name, sp, info = run
    if sp is None:
        print(f"{name}: skipped ({info['reason']})")
    else:
        extra = f", pleiotropy {info['pleiotropy']}" if info["pleiotropy"] != "off" else ""
        miss = f", missing {','.join(info['missing'])}" if info["missing"] else ""
        print(f"{name}: {info['n_sets']} set(s), {info['n_samples']} samples{extra}{miss}")


def _normalise(cohorts) -> list:
    if not isinstance(cohorts, Mapping) or not len(cohorts):
        raise ValueError("cohorts must be a non-empty dict name -> (network, expression)")
    items = []
    for name, v in cohorts.items():
        if isinstance(v, Mapping):
            v = (v.get("network"), v.get("expression"))
        if not isinstance(v, (tuple, list)) or len(v) != 2 or v[0] is None or v[1] is None:
            raise ValueError(f"Cohort {name!r}: give (network, expression) or {{'network': ..., 'expression': ...}}")
        items.append((str(name), (v[0], v[1])))
    return items


def activity_specificity_across_cohorts(cohorts: Mapping, sets: Sequence[Sequence[str]], *, covariates=None,
                                        sample_id: Optional[Callable[[str], str]] = None, pleiotropy: bool = True,
                                        pleiotropy_args: Optional[dict] = None, method: str = "mad", minsize: int = 25,
                                        direction: str = "abs", top: int = 5, min_samples: int = 20, score_minsize: int = 10,
                                        n_jobs: int = 1, cache_dir: Union[str, Path, None] = None, expression_reader: Optional[Callable] = None,
                                        regulon_reader: Optional[Callable] = None, verbose: bool = True) -> MultiActivity:
    """Activity of regulator sets in many cohorts: correlations, partner ranks, covariate adjustment and the pleiotropy correction (needs R).

    Parameters
    ----------
    cohorts : dict
        ``name -> (network, expression)`` (or ``{"network": ..., "expression": ...}``): the network (regulon) of the cohort as the path of
        an ``.rds`` file or a long-format DataFrame, and its expression (genes x samples, normalised, e.g. logCPM) as a path (tab-separated,
        first column the genes; ``expression_reader=`` for other formats) or a DataFrame. The activity of the whole network is inferred
        (``eset_filter=False``) for the partner ranks.
    sets : sequence of sequences of str
        The sets to evaluate, as in :func:`pyrea.activity_specificity`.
    covariates : optional
        Background variables that adjust the correlations and ranks (none by default). Give the name of a registry variable
        (``"absolute_purity"``, see :func:`pyrea.available_covariates`), a **dict of gene lists** ``{"EMT": genes, "Hypoxia": genes}`` that is scored
        per sample from the cohort's expression with :func:`pyrea.score_genesets` (aREA), a DataFrame / Series indexed by sample name for **all**
        cohorts (missing samples count as missing), or a list of these (see :func:`pyrea.resolve_covariates`). Gene list scores are computed
        from the same expression as the activities: a list that contains the regulators or their targets removes what is being tested. ``sample_id`` maps the column names of the expression to this index (e.g. :func:`pyrea.tcga_sample_id`; a
        function for ``n_jobs`` > 1 must be picklable, i.e. defined in a module, not a lambda); if it makes names repeat, the first
        sample is kept (``coverage``).
    pleiotropy : bool
        Also infer the activity of the candidates from a **regulon restricted to the candidates** with VIPER's pleiotropy correction (extra
        activity variant ``"pleiotropy"``, see :meth:`MultiActivity.pleiotropy_effect`); ``pleiotropy_args`` as in
        :func:`pyrea.viper_activity`. If VIPER's R code fails in a cohort the correction is left out there (``coverage``).
    method, minsize :
        Signature method and minimum regulon size of :func:`pyrea.viper_activity`.
    direction, top, min_samples :
        As in :func:`pyrea.activity_specificity`.
    score_minsize : int
        Smallest number of genes of a gene list covariate that must be present in the expression of a cohort.
    n_jobs : int
        Cohorts in parallel in separate processes (each holds one expression matrix and the network, and starts R).
    cache_dir : path, optional
        Result per cohort; key: both files (size, modification time), the sets, all settings and the covariates.
    expression_reader, regulon_reader : callable, optional
        ``reader(path) -> DataFrame`` (must be importable if ``n_jobs`` > 1).

    Returns
    -------
    MultiActivity

    Examples
    --------
    ::

        cohorts = {"BRCA": (net_paths["BRCA"], "TCGA-BRCA-logCPM.tsv"), "STAD": (net_paths["STAD"], "TCGA-STAD-logCPM.tsv")}
        ma = pr.activity_specificity_across_cohorts(cohorts, [("ZCCHC24", "ZEB1")], covariates=["absolute_purity", {"Hypoxia": hypoxia_genes}],
                                                    sample_id=pr.tcga_sample_id, n_jobs=4)
        ma.summary(); ma.meta(); ma.plot_forest("ZCCHC24", "ZEB1", adjusted=True)
    """
    items = _normalise(cohorts)
    sets = _check_sets(sets)
    kw = dict(method=method, minsize=int(minsize), direction=direction, top=int(top), min_samples=int(min_samples),
              score_minsize=int(score_minsize), pleiotropy=bool(pleiotropy), pleiotropy_args=None if pleiotropy_args is None else dict(sorted(pleiotropy_args.items())))
    for part in _flatten(covariates):
        if isinstance(part, (pd.DataFrame, pd.Series)) and part.index.duplicated().any():
            raise ValueError("covariates has duplicated sample names")
    if sample_id is not None and n_jobs and n_jobs > 1:
        try:
            pickle.dumps(sample_id)
        except Exception as exc:
            raise ValueError("sample_id must be picklable for n_jobs > 1 (a function of an installed module such as pyrea.tcga_sample_id; "
                             "a lambda or a function defined in a notebook is not); or use n_jobs=1") from exc
    fn = partial(_run_cohort, sets=sets, kw=kw, covariates=covariates, sample_id=sample_id, cache_dir=cache_dir,
                 expression_reader=expression_reader, regulon_reader=regulon_reader)
    runs = _run_all(items, fn, n_jobs, _say if verbose else None)
    results = {n: sp for n, sp, info in runs if sp is not None}
    skipped = {n: info["reason"] for n, sp, info in runs if sp is None}
    for n, sp, info in runs:
        if sp is None:
            warnings.warn(f"{n}: skipped ({info['reason']}; regulators missing: {info['missing']})")
    if not results:
        raise ValueError("None of the cohorts contains a requested set")
    where = {n: tuple("" if isinstance(x, pd.DataFrame) else str(x) for x in v) for n, v in items}
    coverage = pd.DataFrame([{"cohort": n, "network": where[n][0], "expression": where[n][1], "n_sets": info["n_sets"],
                              "missing": ",".join(info["missing"]), "n_samples": info["n_samples"], "n_dropped": info["n_dropped"],
                              "n_duplicated_samples": info["n_duplicated_samples"], "pleiotropy": info["pleiotropy"]}
                             for n, sp, info in runs])

    def cat(attr):
        out = pd.concat([getattr(sp, attr).assign(cohort=n) for n, sp in results.items()], ignore_index=True)
        return out[["cohort"] + [c for c in out.columns if c != "cohort"]]

    first = next(iter(results.values())).settings
    settings = {k: kw[k] for k in _SETTING_KEYS}
    settings.update(covariates=first["covariates"], sets=[" | ".join(s) for s in sets], n_cohorts=len(items))
    return MultiActivity(results=results, skipped=skipped, coverage=coverage, pairs=cat("pairs"), table=cat("table"),
                         extension=cat("extension"), settings=settings)

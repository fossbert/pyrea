"""Specificity of regulator sets across many networks (e.g. one ARACNe network per tumour type).

:func:`specificity_across_networks` runs :func:`pyrea.regulon_specificity` on a list of networks, one network at a time in memory
(or ``n_jobs`` in parallel processes), and collects the result in a :class:`MultiSpecificity`: tables with the column ``network``, a
summary of in how many networks a set is more specific than regulators of the same size, and the networks x sets figure. Networks,
names, cache and merging work as in :func:`pyrea.overlap_across_networks`.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .regulon_overlap_multi import _bh, _cache_file, _normalise, _read, _run_all
from .regulon_specificity import _check_sets, regulon_specificity

__all__ = ["specificity_across_networks", "MultiSpecificity"]

_KEYS = ["set", "signs"]
_SETTING_KEYS = ("tail", "mor_min", "min_likelihood", "min_size", "n_random", "tol", "min_pool", "seed")


def _add_q(table: pd.DataFrame) -> pd.DataFrame:
    table = table.drop(columns=[c for c in ("q_random_all",) if c in table.columns])
    table["q_random_all"] = _bh(table["p_random"])
    return table


@dataclass
class MultiSpecificity:
    """Result of :func:`specificity_across_networks`.

    Attributes
    ----------
    results : dict
        Network name -> :class:`pyrea.RegulonSpecificity`.
    skipped : dict
        Network name -> reason, for the networks that contain none of the requested sets.
    coverage : pd.DataFrame
        One row per network: ``path``, ``n_sets`` evaluated, the regulators ``missing`` in it and ``n_regulators`` in the reference.
    table, extension : pd.DataFrame
        The ``table`` and ``extension`` of all networks with ``network`` as first column (see :class:`pyrea.RegulonSpecificity`).
        ``table`` has ``q_random_all`` in addition: Benjamini-Hochberg of ``p_random`` over **all** rows of all networks.
    settings : dict
        The settings of the analysis and the sets requested.

    Methods
    -------
    ``summary()``, ``matrix()``, ``hits()``: tables. ``merge()``, ``concat()``: combine analyses of different networks by name.
    ``export()``. ``plot_matrix()``, ``add_hits()``: networks x sets figure.
    """

    results: dict
    skipped: dict
    coverage: pd.DataFrame
    table: pd.DataFrame
    extension: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"MultiSpecificity({len(self.results)} network(s), {len(self.skipped)} skipped, "
                f"{self.table['set'].nunique() if len(self.table) else 0} set(s), tail={self.settings.get('tail')!r})")

    # ------------------------------------------------------------------ combining
    @classmethod
    def concat(cls, parts: Sequence["MultiSpecificity"]) -> "MultiSpecificity":
        """Combine analyses of **different networks** (matched by name): same ``tail``, ``mor_min``, ``min_likelihood``, ``min_size``,
        ``n_random``, ``tol``, ``min_pool`` and ``seed`` and no repeated network name are required (an error otherwise). The sets may
        differ. ``q_random_all`` is recomputed over all rows of the combined table."""
        parts = list(parts)
        if not parts:
            raise ValueError("Nothing to concatenate")
        ref = parts[0].settings
        for i, m in enumerate(parts[1:], 1):
            diff = [k for k in _SETTING_KEYS if m.settings.get(k) != ref.get(k)]
            if diff:
                raise ValueError(f"Analysis {i} differs in {diff} from the first one; combine only analyses with the same settings")
        names = [n for m in parts for n in list(m.results) + list(m.skipped)]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"Network names occur in more than one analysis: {dup}; rename them (names= / dict keys)")
        settings = dict(ref)
        settings["sets"] = list(dict.fromkeys(s for m in parts for s in m.settings.get("sets", [])))
        settings["n_networks"] = sum(m.settings.get("n_networks", len(m.results) + len(m.skipped)) for m in parts)
        return cls(results={n: r for m in parts for n, r in m.results.items()},
                   skipped={n: w for m in parts for n, w in m.skipped.items()},
                   coverage=pd.concat([m.coverage for m in parts], ignore_index=True),
                   table=_add_q(pd.concat([m.table for m in parts], ignore_index=True)),
                   extension=pd.concat([m.extension for m in parts], ignore_index=True), settings=settings)

    def merge(self, *others: "MultiSpecificity") -> "MultiSpecificity":
        """This analysis combined with others of different networks (see :meth:`concat`)."""
        return MultiSpecificity.concat([self, *others])

    # ------------------------------------------------------------------ tables
    def _default_sign(self) -> str:
        return "-" if self.settings.get("tail") == "negative" else "+"

    def _sign_mask(self, signs: Optional[str]) -> pd.Series:
        t = self.table
        if signs is None:
            d = self._default_sign()
            return t["signs"].map(lambda s: set(s.split("/")) == {d})
        return t["signs"] == signs

    def hits(self, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3) -> pd.Series:
        """Boolean per row of ``table``: ``q_random_all`` < ``q``, ``fold`` >= ``fold_min`` and ``overlap`` >= ``min_overlap``: the set
        overlaps more than size-matched random sets of other regulators."""
        t = self.table
        return (t["q_random_all"] < q) & (t["fold"] >= fold_min) & (t["overlap"] >= min_overlap)

    def summary(self, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, top_frac: float = 0.05) -> pd.DataFrame:
        """Per set and tail combination: in how many networks it was evaluated (``n_networks``) and is a hit (``n_hit``, see
        :meth:`hits`), ``frac_hit``, the median ``fold``, the median fold of the matched random sets (``median_null_fold``), the median
        ``percentile`` against them, the median ``ext_percentile_max`` (worst partner rank of a member, as a fraction of the
        candidates; small = specific), the number of networks in which every member is the best partner of the others
        (``n_mutual_best``), the number of networks in which every member is within the best ``top_frac`` of the partners of the
        others (``n_top``; no multiple-testing floor, unlike the hits) and the networks with a hit. The networks are not independent
        (related tumour types)."""
        t = self.table.assign(hit=self.hits(q, fold_min, min_overlap))
        rows = []
        with warnings.catch_warnings():                                       # all-NaN medians (sets with an empty tail)
            warnings.simplefilter("ignore", RuntimeWarning)
            for key, d in t.groupby(_KEYS, sort=False):
                rows.append(dict(zip(_KEYS, key), n_networks=len(d), n_hit=int(d["hit"].sum()), frac_hit=float(d["hit"].mean()),
                                 median_fold=float(d["fold"].median()), median_null_fold=float(d["null_median_fold"].median()),
                                 median_percentile=float(d["percentile"].median()),
                                 median_ext_percentile=float(d["ext_percentile_max"].median()),
                                 n_mutual_best=int(d["mutual_best"].sum()),
                                 n_top=int((d["ext_percentile_max"] <= top_frac).sum()), networks_hit=";".join(d.loc[d["hit"], "network"])))
        cols = _KEYS + ["n_networks", "n_hit", "frac_hit", "median_fold", "median_null_fold", "median_percentile",
                        "median_ext_percentile", "n_mutual_best", "n_top", "networks_hit"]
        return pd.DataFrame(rows, columns=cols).sort_values(["frac_hit", "median_percentile"], ascending=False, ignore_index=True)

    def matrix(self, value: str = "percentile", signs: Optional[str] = None) -> pd.DataFrame:
        """Networks (rows) x sets (columns) of one column of ``table`` for one tail combination. ``signs=None``: all members in the
        positive tail (negative tail if ``tail="negative"``); otherwise the pattern, e.g. ``"+/-"``. NaN where a set was not evaluated."""
        t = self.table
        if value not in t.columns:
            raise ValueError(f"Unknown value {value!r}; columns: {list(t.columns)}")
        t = t[self._sign_mask(signs)]
        if t.empty:
            raise ValueError(f"Tail combination {signs!r} not in the table (tail={self.settings.get('tail')!r})")
        mat = t.pivot(index="network", columns="set", values=value)
        return mat.reindex(index=[n for n in self.results if n in mat.index], columns=list(dict.fromkeys(t["set"])))

    def hit_cells(self, signs: Optional[str] = None, **kw) -> list:
        """(network, set) of the hits for one tail combination (for :func:`add_hits`)."""
        t = self.table[self._sign_mask(signs) & self.hits(**kw)]
        return list(zip(t["network"], t["set"]))

    # ------------------------------------------------------------------ export and figures
    def export(self, path: Union[str, Path]) -> Path:
        """Write one Excel workbook (``.xlsx``) or a folder of CSV files with ``settings``, ``coverage``, ``summary``, ``table`` and
        ``extension``."""
        path = Path(path)
        tables = {"settings": pd.DataFrame([{k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in self.settings.items()}]),
                  "coverage": self.coverage, "summary": self.summary(), "table": self.table, "extension": self.extension}
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for name, df in tables.items():
                    df.to_excel(xw, sheet_name=name[:31], index=False)
        else:
            path.mkdir(parents=True, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(path / f"{name}.csv", index=False)
        return path

    def plot_matrix(self, value: str = "percentile", signs: Optional[str] = None, ax=None, cmap: str = "Blues",
                    title: Optional[str] = None):
        """Networks x sets heatmap of one column of ``table``; hits are marked afterwards with :meth:`add_hits`."""
        from .regulon_specificity_plot import plot_matrix
        return plot_matrix(self, value, signs, ax, cmap, title)

    def add_hits(self, ax=None, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, marker: str = "*", **text_kwargs):
        """After :meth:`plot_matrix`: mark the cells that are hits (:meth:`hits`)."""
        from .regulon_specificity_plot import add_hits
        return add_hits(self, ax, q, fold_min, min_overlap, marker, **text_kwargs)


# ---------------------------------------------------------------------- running the networks
def _run_network(name, item, sets, kw, cache_dir, reader):
    settings = (tuple(sets), tuple((k, (hashlib.sha1(repr(sorted(v)).encode()).hexdigest() if k == "universe" and v is not None else v))
                                   for k, v in sorted(kw.items())), "specificity")
    f = _cache_file(cache_dir, name, item, settings)
    if f is not None and f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    reg = _read(item, reader)
    have = set(reg["source"].astype(str))
    present = [s for s in sets if all(t in have for t in s)]
    missing = sorted({t for s in sets for t in s} - have)
    if not present:
        out = (name, None, {"n_sets": 0, "missing": missing, "reason": "none of the requested sets is in the network"})
    else:
        sp = regulon_specificity(reg, present, **kw)
        out = (name, sp, {"n_sets": len(present), "missing": missing, "reason": ""})
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(out, fh)
    return out


def _say(run) -> None:
    name, sp, info = run
    if sp is None:
        print(f"{name}: skipped ({info['reason']})")
    else:
        miss = f", missing {','.join(info['missing'])}" if info["missing"] else ""
        print(f"{name}: {info['n_sets']} set(s), {sp.settings['n_regulators']['+']} regulators in the reference{miss}")


def specificity_across_networks(networks, sets: Sequence[Sequence[str]], *, names: Optional[Sequence[str]] = None,
                                tail: str = "positive", mor_min: float = 0.0, min_likelihood: float = 0.0,
                                universe: Optional[Iterable[str]] = None, min_size: int = 25, n_random: int = 5000, tol: float = 0.2,
                                min_pool: int = 50, seed: int = 0, top: int = 5, n_jobs: int = 1,
                                cache_dir: Union[str, Path, None] = None, reader: Optional[Callable] = None,
                                verbose: bool = True) -> MultiSpecificity:
    """Is a set of regulators more closely linked than other regulators, in many networks?

    Runs :func:`pyrea.regulon_specificity` on every network. ``networks`` (paths of ``.rds`` files, or a dict ``name -> path``), ``names``,
    ``n_jobs``, ``cache_dir`` and ``reader`` work as in :func:`pyrea.overlap_across_networks` (one network in memory at a time; cache
    keyed on file, sets and settings). ``sets``: the regulator sets to evaluate, e.g. ``[("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")]``.
    A network lacking some regulators of a set is evaluated for the other sets (``coverage``); one that contains none is skipped.
    All other parameters are those of :func:`pyrea.regulon_specificity`.

    Returns
    -------
    MultiSpecificity

    Examples
    --------
    ::

        ms = pr.specificity_across_networks(net_paths, [("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")], n_random=5000)
        ms.summary()
        ms.plot_matrix("percentile"); ms.add_hits()
    """
    items = _normalise(networks, names)
    sets = _check_sets(sets, tail)
    kw = dict(tail=tail, mor_min=float(mor_min), min_likelihood=float(min_likelihood),
              universe=None if universe is None else sorted(map(str, universe)), min_size=int(min_size), n_random=int(n_random),
              tol=float(tol), min_pool=int(min_pool), seed=int(seed), top=int(top))
    runs = _run_all(items, partial(_run_network, sets=sets, kw=kw, cache_dir=cache_dir, reader=reader), n_jobs,
                    _say if verbose else None)
    results = {n: sp for n, sp, info in runs if sp is not None}
    skipped = {n: info["reason"] for n, sp, info in runs if sp is None}
    for n, sp, info in runs:
        if sp is None:
            warnings.warn(f"{n}: skipped ({info['reason']}; regulators missing: {info['missing']})")
    if not results:
        raise ValueError("None of the networks contains a requested set")
    where = {n: ("" if isinstance(x, pd.DataFrame) else str(x)) for n, x in items}
    coverage = pd.DataFrame([{"network": n, "path": where[n], "n_sets": info["n_sets"], "missing": ",".join(info["missing"]),
                              "n_regulators": sp.settings["n_regulators"]["+"] if sp is not None else np.nan} for n, sp, info in runs])

    def cat(attr):
        out = pd.concat([getattr(sp, attr).assign(network=n) for n, sp in results.items()], ignore_index=True)
        return out[["network"] + [c for c in out.columns if c != "network"]]

    first = next(iter(results.values())).settings
    settings = {k: first[k] for k in _SETTING_KEYS}
    settings.update(sets=[" | ".join(s) for s in sets], n_networks=len(items))
    return MultiSpecificity(results=results, skipped=skipped, coverage=coverage, table=_add_q(cat("table")), extension=cat("extension"),
                            settings=settings)

"""Overlap of regulons across many networks (e.g. one ARACNe network per tumour type).

:func:`overlap_across_networks` runs :func:`pyrea.regulon_overlap` on a list of networks (paths to ``.rds`` files), one network at a
time in memory, and collects everything in a :class:`MultiOverlap`: one long table (column ``network``), the shared targets of all
networks for export, a per-pair summary of how many networks show the overlap, and the targets that recur across networks.

The pairs to test are chosen as in :func:`pyrea.regulon_overlap`: all pairs of ``tfs``, one ``focus`` regulator against the others, or
explicit ``pairs``. A network that lacks some of the regulators is analysed for the pairs it covers; one that covers no pair is skipped
(``skipped``, ``coverage``).
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from .regulon_overlap import _SIGNS, RegulonOverlap, _resolve_pairs, regulon_overlap

__all__ = ["overlap_across_networks", "MultiOverlap"]

_KEYS = ["tf1", "tf2", "sign1", "sign2"]


def _bh(p: pd.Series) -> pd.Series:
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.sum() > 1:
        q[ok] = false_discovery_control(p[ok].to_numpy(), method="bh")
    elif ok.sum() == 1:
        q[ok] = p[ok]
    return q


def _add_q_all(table: pd.DataFrame) -> pd.DataFrame:
    """``q_hyper_all`` / ``q_perm_all``: Benjamini-Hochberg over all rows of the table (recomputed from the p-values)."""
    table = table.drop(columns=[c for c in ("q_hyper_all", "q_perm_all") if c in table.columns])
    for p in [c for c in ("p_hyper", "p_perm") if c in table.columns]:
        table["q_" + p[2:] + "_all"] = _bh(table[p])
    return table


def _pair_label(df: pd.DataFrame) -> pd.Series:
    return df["tf1"] + " | " + df["tf2"]


@dataclass
class MultiOverlap:
    """Result of :func:`overlap_across_networks`.

    Attributes
    ----------
    results : dict
        Network name -> :class:`pyrea.RegulonOverlap` (small: only the regulators asked for), for the figures, ``enrich`` and
        ``split_activity`` of a single network.
    skipped : dict
        Network name -> reason, for the networks that cover none of the requested pairs.
    coverage : pd.DataFrame
        One row per network: ``path`` of the file (empty for a DataFrame), ``n_pairs`` tested, the regulators ``missing`` in it and
        ``universe_size``.
    table : pd.DataFrame
        The ``table`` of all networks (see :class:`pyrea.RegulonOverlap`) with ``network`` as first column and, in addition
        to the per-network ``q_hyper`` / ``q_perm``, ``q_hyper_all`` / ``q_perm_all``: Benjamini-Hochberg over **all** rows of all
        networks.
    pairs, sizes : pd.DataFrame
        ``pairs`` and ``sizes`` of all networks, with ``network``.
    shared_targets : pd.DataFrame
        The shared targets of all pairs in all networks (long format, ``network`` first), for export.
    settings : dict
        ``tail``, ``mor_min``, ``min_likelihood``, ``n_perm``, ``seed`` and the pairs requested.

    Methods
    -------
    ``summary()``, ``matrix()``, ``recurrent_targets()``: tables. ``merge()``, ``concat()``: combine analyses of different networks by name. ``collect()``, ``enrich()``, ``export()``: gene lists of the targets
    that recur across networks. ``plot_matrix()``, ``add_hits()``: networks x pairs figure.
    """

    results: dict
    skipped: dict
    coverage: pd.DataFrame
    table: pd.DataFrame
    pairs: pd.DataFrame
    sizes: pd.DataFrame
    shared_targets: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"MultiOverlap({len(self.results)} network(s), {len(self.skipped)} skipped, "
                f"{self.table[['tf1', 'tf2']].drop_duplicates().shape[0] if len(self.table) else 0} pair(s), "
                f"tail={self.settings.get('tail')!r})")

    # ------------------------------------------------------------------ combining
    @classmethod
    def concat(cls, parts: Sequence["MultiOverlap"]) -> "MultiOverlap":
        """Combine analyses of **different networks** (matched by name) into one, e.g. cohorts added later or run separately.

        The analyses must use the same ``tail``, ``mor_min``, ``min_likelihood``, ``n_perm`` and ``seed`` and must not share a network name
        (an error otherwise). They may test different pairs; ``summary()`` then counts the networks in which each pair was tested. The
        per-network results are kept, ``q_hyper_all`` / ``q_perm_all`` are **recomputed over all rows** of the combined table (the
        correction depends on the number of tests), and the order of the networks is that of ``parts``.
        """
        parts = list(parts)
        if not parts:
            raise ValueError("Nothing to concatenate")
        keys = ("tail", "mor_min", "min_likelihood", "n_perm", "seed")
        ref = parts[0].settings
        for i, m in enumerate(parts[1:], 1):
            diff = [k for k in keys if m.settings.get(k) != ref.get(k)]
            if diff:
                raise ValueError(f"Analysis {i} differs in {diff} from the first one; combine only analyses with the same settings "
                                 f"({', '.join(f'{k}={ref.get(k)!r}' for k in diff)} vs "
                                 f"{', '.join(f'{k}={m.settings.get(k)!r}' for k in diff)})")
        names = [n for m in parts for n in list(m.results) + list(m.skipped)]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"Network names occur in more than one analysis: {dup}; rename them (names= / dict keys)")

        def cat(attr):
            out = pd.concat([getattr(m, attr) for m in parts], ignore_index=True)
            return out
        results = {n: ov for m in parts for n, ov in m.results.items()}
        table = _add_q_all(pd.concat([m.table for m in parts], ignore_index=True))
        settings = dict(ref)
        settings["pairs"] = list(dict.fromkeys(p for m in parts for p in m.settings.get("pairs", [])))
        settings["n_networks"] = sum(m.settings.get("n_networks", len(m.results) + len(m.skipped)) for m in parts)
        return cls(results=results, skipped={n: w for m in parts for n, w in m.skipped.items()}, coverage=cat("coverage"),
                   table=table, pairs=cat("pairs"), sizes=cat("sizes"), shared_targets=cat("shared_targets"), settings=settings)

    def merge(self, *others: "MultiOverlap") -> "MultiOverlap":
        """This analysis combined with others of different networks (see :meth:`concat`)."""
        return MultiOverlap.concat([self, *others])

    # ------------------------------------------------------------------ tables
    def _default_signs(self, signs):
        return tuple(signs) if signs is not None else _SIGNS[self.settings.get("tail", "positive")][0]

    def _qcol(self, p: str = "auto") -> str:
        if p not in ("auto", "perm", "hyper"):
            raise ValueError("p must be 'auto', 'perm' or 'hyper'")
        has_perm = "q_perm_all" in self.table.columns
        if p == "perm" and not has_perm:
            raise ValueError("The analysis was run without permutations (n_perm=0)")
        return "q_perm_all" if (p == "perm" or (p == "auto" and has_perm)) else "q_hyper_all"

    def hits(self, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, p: str = "auto") -> pd.Series:
        """Boolean per row of ``table``: ``q_*_all`` < ``q``, ``fold`` >= ``fold_min`` and ``overlap`` >= ``min_overlap``.
        ``p="auto"`` uses the permutation q-value if there is one, else the hypergeometric (``"perm"`` / ``"hyper"`` force)."""
        t = self.table
        return (t[self._qcol(p)] < q) & (t["fold"] >= fold_min) & (t["overlap"] >= min_overlap)

    def summary(self, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, p: str = "auto") -> pd.DataFrame:
        """Per pair and tail combination: in how many networks it was tested (``n_networks``) and is a hit (``n_hit``: see
        :meth:`hits`), ``frac_hit``, the median ``fold``, ``jaccard`` and ``overlap`` over the networks, the smallest ``q_*_all``
        and the networks with a hit. The networks are not independent (related tumour types, overlapping samples), so this is a
        description, not a test of consistency."""
        t = self.table.assign(hit=self.hits(q, fold_min, min_overlap, p))
        qc = self._qcol(p)
        rows = []
        for key, d in t.groupby(_KEYS, sort=False):
            rows.append(dict(zip(_KEYS, key), n_networks=len(d), n_hit=int(d["hit"].sum()), frac_hit=float(d["hit"].mean()),
                             median_fold=float(d["fold"].median()), median_jaccard=float(d["jaccard"].median()),
                             median_overlap=float(d["overlap"].median()), min_q=float(d[qc].min()),
                             networks_hit=";".join(d.loc[d["hit"], "network"])))
        out = pd.DataFrame(rows, columns=_KEYS + ["n_networks", "n_hit", "frac_hit", "median_fold", "median_jaccard",
                                                  "median_overlap", "min_q", "networks_hit"])
        return out.sort_values(["frac_hit", "median_fold"], ascending=False, ignore_index=True)

    def matrix(self, value: str = "fold", signs: Optional[tuple] = None) -> pd.DataFrame:
        """Networks (rows) x pairs (columns ``"TF1 | TF2"``) of one column of ``table`` for one tail combination
        (default: the first of the analysis, e.g. ``("+", "+")``). NaN where a pair was not tested in a network."""
        signs = self._default_signs(signs)
        t = self.table
        if value not in t.columns:
            raise ValueError(f"Unknown value {value!r}; columns: {list(t.columns)}")
        t = t[(t["sign1"] == signs[0]) & (t["sign2"] == signs[1])]
        if t.empty:
            raise ValueError(f"Sign combination {signs} not in the table (tail={self.settings.get('tail')!r})")
        mat = t.assign(pair=_pair_label(t)).pivot(index="network", columns="pair", values=value)
        order = list(dict.fromkeys(_pair_label(t)))
        return mat.reindex(index=[n for n in self.results if n in mat.index], columns=order)

    def recurrent_targets(self, min_networks: int = 2, signs: Optional[tuple] = None) -> pd.DataFrame:
        """Shared targets that recur: per pair, the genes that are shared in the tail combination ``signs`` in at least
        ``min_networks`` networks. Columns ``tf1``, ``tf2``, ``gene``, ``n_networks``, ``n_tested`` (networks in which the pair
        was tested), ``frac``, ``mean_likelihood1`` / ``mean_likelihood2`` and ``networks``; most recurrent first."""
        signs = self._default_signs(signs)
        st = self.shared_targets
        st = st[(st["sign1"] == signs[0]) & (st["sign2"] == signs[1])]
        tested = self.table.drop_duplicates(["tf1", "tf2", "network"]).groupby(["tf1", "tf2"]).size().rename("n_tested")
        cols = ["tf1", "tf2", "gene", "n_networks", "n_tested", "frac", "mean_likelihood1", "mean_likelihood2", "networks"]
        if st.empty:
            return pd.DataFrame(columns=cols)
        g = st.groupby(["tf1", "tf2", "gene"], sort=False)
        out = g.agg(n_networks=("network", "nunique"), mean_likelihood1=("likelihood1", "mean"),
                    mean_likelihood2=("likelihood2", "mean"), networks=("network", lambda x: ";".join(sorted(set(x))))).reset_index()
        out = out.join(tested, on=["tf1", "tf2"])
        out["frac"] = out["n_networks"] / out["n_tested"]
        out = out[out["n_networks"] >= min_networks]
        return out[cols].sort_values(["tf1", "tf2", "n_networks", "mean_likelihood1"], ascending=[True, True, False, False],
                                     ignore_index=True)

    # ------------------------------------------------------------------ gene lists, enrichment, export
    def collect(self, min_networks: int = 2, signs: Optional[tuple] = None, into=None, per_network: bool = False):
        """The recurrent shared targets (:meth:`recurrent_targets`) as a :class:`pyrea.GeneListCollection`, one list per pair
        (``"TF1&TF2 (+/+) in >= k networks"``), for later export. ``per_network=True`` adds the lists of every network too
        (prefixed with its name). Returns the collection (a new one if ``into`` is not given)."""
        from .genelists import GeneListCollection
        signs = self._default_signs(signs)
        col = GeneListCollection() if into is None else into
        rec = self.recurrent_targets(min_networks, signs)
        for (a, b), d in rec.groupby(["tf1", "tf2"], sort=False):
            col.add(f"{a}&{b} ({signs[0]}/{signs[1]}) in >= {min_networks} networks", d["gene"], source="across networks",
                    tfs=f"{a},{b}", signs=f"{signs[0]}/{signs[1]}", tail=self.settings.get("tail"), min_networks=min_networks,
                    n_tested=int(d["n_tested"].iloc[0]), n_genes=len(d))
        if per_network:
            for name, ov in self.results.items():
                ov.collect(into=col, label=name, signs=signs)
        return col

    def enrich(self, genesets, min_networks: int = 2, signs: Optional[tuple] = None, universe: Optional[Iterable[str]] = None,
               min_size: int = 5, max_size: int = 500, min_overlap: int = 2, species: str = "human") -> pd.DataFrame:
        """Over-representation of gene sets in the recurrent shared targets of every pair (see :func:`pyrea.gene_set_enrichment`;
        ``genesets`` may be the name of a bundled collection). The default universe of a pair is the intersection of the target
        universes of the networks in which it was tested. Columns as in :meth:`pyrea.RegulonOverlap.enrich`."""
        from .genelists import gene_set_enrichment, resolve_genesets
        resolved, coll = resolve_genesets(genesets, species)
        signs = self._default_signs(signs)
        rec = self.recurrent_targets(min_networks, signs)
        frames = []
        for (a, b), d in rec.groupby(["tf1", "tf2"], sort=False):
            if universe is None:
                nets = self.table.loc[(self.table["tf1"] == a) & (self.table["tf2"] == b), "network"].unique()
                uni = set.intersection(*(set(self.results[n].universe_genes) for n in nets))
            else:
                uni = universe
            e = gene_set_enrichment(d["gene"], resolved, universe=uni, min_size=min_size, max_size=max_size, min_overlap=min_overlap)
            if coll is not None:
                e.insert(1, "collection", e["term"].map(coll))
            e.insert(0, "target_set", f"{a}&{b} ({signs[0]}/{signs[1]}) in >= {min_networks} networks")
            frames.append(e)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["target_set", "term"])

    def export(self, path: Union[str, Path], min_networks: int = 2, signs: Optional[tuple] = None, collection=None,
               enrichment: Optional[pd.DataFrame] = None) -> Path:
        """Write one Excel workbook (``.xlsx``), a GMT file (``.gmt``, gene lists only) or a folder of CSV files with the sheets
        ``settings``, ``coverage``, ``summary``, ``table``, ``pairs``, ``sizes``, ``shared_targets`` (all networks), ``recurrent``,
        ``lists`` / ``genes`` (``collection``, default ``collect(min_networks, signs)``) and, if given, ``enrichment``."""
        col = self.collect(min_networks, signs) if collection is None else collection
        extra = {"settings": pd.DataFrame([{k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in self.settings.items()}]),
                 "coverage": self.coverage, "summary": self.summary(), "table": self.table, "pairs": self.pairs, "sizes": self.sizes,
                 "shared_targets": self.shared_targets, "recurrent": self.recurrent_targets(min_networks, signs)}
        if enrichment is not None:
            extra["enrichment"] = enrichment
        return col.export(path, extra=extra)

    # ------------------------------------------------------------------ figures: see pyrea.regulon_overlap_plot
    def plot_matrix(self, value: str = "fold", signs: Optional[tuple] = None, ax=None, cmap: str = "Blues",
                    title: Optional[str] = None):
        """Networks x pairs heatmap of one column of ``table``; hits are marked afterwards with :meth:`add_hits`."""
        from .regulon_overlap_plot import plot_matrix
        return plot_matrix(self, value, signs, ax, cmap, title)

    def add_hits(self, ax=None, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, p: str = "auto",
                 marker: str = "*", **text_kwargs):
        """After :meth:`plot_matrix`: mark the cells that are hits (:meth:`hits`)."""
        from .regulon_overlap_plot import add_hits
        return add_hits(self, ax, q, fold_min, min_overlap, p, marker, **text_kwargs)


# ---------------------------------------------------------------------- running the networks
def _read(item, reader):
    if isinstance(item, pd.DataFrame):
        return item
    if reader is None:
        from .rbackend.regulon import read_regulon_rds
        reader = read_regulon_rds
    return reader(item)


def _cache_file(cache_dir, name, item, settings) -> Optional[Path]:
    if cache_dir is None or isinstance(item, pd.DataFrame):
        return None
    st = Path(item).stat()
    key = hashlib.sha1(repr((name, str(Path(item).resolve()), st.st_size, st.st_mtime_ns, settings)).encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return Path(cache_dir) / f"{safe}-{key}.pkl"


def _run_network(name, item, requested, kw, cache_dir, reader):
    """One network: returns (name, RegulonOverlap or None, info). ``requested`` is the list of pairs."""
    # the universe may be large: its digest goes into the cache key
    settings = (tuple(requested), tuple((k, (hashlib.sha1(repr(sorted(v)).encode()).hexdigest() if k == "universe" and v is not None else v))
                                       for k, v in sorted(kw.items())))
    f = _cache_file(cache_dir, name, item, settings)
    if f is not None and f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    reg = _read(item, reader)
    have = set(reg["source"].astype(str))
    present = [p for p in requested if p[0] in have and p[1] in have]
    missing = sorted({t for p in requested for t in p} - have)
    if not present:
        out = (name, None, {"n_pairs": 0, "missing": missing, "reason": "none of the requested pairs is in the network"})
    else:
        ov = regulon_overlap(reg, pairs=present, **kw)
        out = (name, ov, {"n_pairs": len(present), "missing": missing, "reason": ""})
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(out, fh)
    return out


def _short_names(paths: Sequence) -> list:
    """File names without extension; if they repeat (``BRCA/ARACNe/FullReg.rds``, ``GBM/FullReg.rds``) the folders that tell the
    files apart: the path parts left after removing the leading parts and the trailing parts that all files share."""
    stems = [Path(p).stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    parts = [list(Path(p).with_suffix("").parts) for p in paths]
    while all(len(q) > 1 for q in parts) and len({q[0] for q in parts}) == 1:
        parts = [q[1:] for q in parts]
    while all(len(q) > 1 for q in parts) and len({q[-1] for q in parts}) == 1:
        parts = [q[:-1] for q in parts]
    return ["/".join(q) for q in parts]


def _normalise(networks, names) -> list:
    if isinstance(networks, Mapping):
        if names is not None:
            raise ValueError("names is only for a sequence; a mapping carries its names")
        items = list(networks.items())
    else:
        seq = [networks] if isinstance(networks, (str, Path, pd.DataFrame)) else list(networks)
        if names is None:
            if any(isinstance(x, pd.DataFrame) for x in seq):
                raise ValueError("DataFrames need names (names=[...] or a mapping name -> network)")
            names = _short_names(seq)
        if len(names) != len(seq):
            raise ValueError("names and networks differ in length")
        items = list(zip(names, seq))
    if len({n for n, _ in items}) != len(items):
        raise ValueError("Network names must be unique (give names= if the file names repeat)")
    if not items:
        raise ValueError("No networks given")
    return [(str(n), x) for n, x in items]


def overlap_across_networks(networks, tfs: Optional[Sequence[str]] = None, *, focus: Optional[str] = None,
                            pairs: Optional[Sequence[tuple]] = None, names: Optional[Sequence[str]] = None,
                            tail: str = "positive", mor_min: float = 0.0, min_likelihood: float = 0.0,
                            universe: Optional[Iterable[str]] = None, n_perm: int = 1000, seed: int = 0, n_jobs: int = 1,
                            cache_dir: Union[str, Path, None] = None, reader: Optional[Callable] = None,
                            verbose: bool = True) -> MultiOverlap:
    """Overlap of regulons by tail in many networks.

    Parameters
    ----------
    networks : sequence of paths, or mapping name -> path
        One network each: paths to ``.rds`` files (read with :func:`pyrea.read_regulon_rds`; ``reader=`` reads other formats) or
        long-format regulon DataFrames (then ``names=`` or a mapping is needed). The networks are read one at a time (``n_jobs``
        at a time in parallel), analysed and released, so only the small results stay in memory. The name of a network is the file
        name without extension (if that repeats, the folders that tell the files apart, e.g. ``"BRCA/ARACNe"``) unless ``names`` /
        a mapping gives one.
    tfs, focus, pairs :
        Which pairs to test, as in :func:`pyrea.regulon_overlap`: all pairs of ``tfs``; ``focus`` against the other ``tfs``; or
        exactly ``pairs``. Pairs with a regulator that a network lacks are left out for that network (``coverage``).
    tail, mor_min, min_likelihood, universe, n_perm, seed :
        As in :func:`pyrea.regulon_overlap`. ``universe`` (default: the targets of each network) is applied to every network.
    n_jobs : int
        Networks analysed in parallel in separate processes (each process reads its own file and holds one network in memory; R
        is started in each of them). The permutations dominate the run time; 1 = sequentially.
    cache_dir : path, optional
        Cache of the result per network. The key is the file (path, size, modification time) and all settings including the
        pairs, so changed settings or files are recomputed; a new set of pairs recomputes the networks.
    reader : callable, optional
        ``reader(path) -> DataFrame`` for files that are not ``.rds`` (must be importable if ``n_jobs`` > 1).

    Returns
    -------
    MultiOverlap

    Examples
    --------
    ::

        paths = sorted(Path("networks").glob("*.rds"))
        mo = pr.overlap_across_networks(paths, ["ZCCHC24", "ZEB1", "ZEB2", "SNAI2"], focus="ZCCHC24", n_perm=1000)
        mo.summary()                    # in how many networks is each pair a hit?
        mo.matrix("fold")               # networks x pairs
        mo.collect(min_networks=5).export("results/recurrent.gmt")
    """
    items = _normalise(networks, names)
    tf_all, requested = _resolve_pairs(tfs, focus, pairs)
    kw = dict(tail=tail, mor_min=float(mor_min), min_likelihood=float(min_likelihood),
              universe=None if universe is None else sorted(map(str, universe)), n_perm=int(n_perm), seed=int(seed))

    if n_jobs is None or n_jobs <= 1 or len(items) == 1:
        runs = []
        for name, item in items:
            runs.append(_run_network(name, item, requested, kw, cache_dir, reader))
            if verbose:
                _say(runs[-1])
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:     # spawn: R / rpy2 must not be forked
            futs = [ex.submit(_run_network, name, item, requested, kw, cache_dir, reader) for name, item in items]
            runs = []
            for fu in futs:
                runs.append(fu.result())
                if verbose:
                    _say(runs[-1])

    results = {n: ov for n, ov, info in runs if ov is not None}
    skipped = {n: info["reason"] for n, ov, info in runs if ov is None}
    for n, ov, info in runs:
        if ov is None:
            warnings.warn(f"{n}: skipped ({info['reason']}; regulators missing: {info['missing']})")
    if not results:
        raise ValueError("None of the networks contains a requested pair")
    where = {n: ("" if isinstance(x, pd.DataFrame) else str(x)) for n, x in items}
    coverage = pd.DataFrame([{"network": n, "path": where[n], "n_pairs": info["n_pairs"], "missing": ",".join(info["missing"]),
                              "universe_size": ov.universe_size if ov is not None else np.nan} for n, ov, info in runs])

    def cat(attr):
        out = pd.concat([getattr(ov, attr).reset_index(drop=True).assign(network=n) for n, ov in results.items()], ignore_index=True)
        return out[["network"] + [c for c in out.columns if c != "network"]]

    table = _add_q_all(cat("table"))
    sizes = pd.concat([ov.sizes.rename_axis("tf").reset_index().assign(network=n) for n, ov in results.items()], ignore_index=True)
    sizes = sizes[["network"] + [c for c in sizes.columns if c != "network"]]
    settings = {"tail": tail, "mor_min": float(mor_min), "min_likelihood": float(min_likelihood), "n_perm": int(n_perm),
                "seed": int(seed), "pairs": [f"{a}|{b}" for a, b in requested], "n_networks": len(items)}
    return MultiOverlap(results=results, skipped=skipped, coverage=coverage, table=table, pairs=cat("pairs"), sizes=sizes,
                        shared_targets=cat("shared_targets"), settings=settings)


def _say(run) -> None:
    name, ov, info = run
    if ov is None:
        print(f"{name}: skipped ({info['reason']})")
    else:
        miss = f", missing {','.join(info['missing'])}" if info["missing"] else ""
        print(f"{name}: {info['n_pairs']} pair(s), universe {ov.universe_size}{miss}")

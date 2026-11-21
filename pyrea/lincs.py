"""Regulator activity in LINCS L1000 signatures: overexpression (OE), CRISPR knockout (XPR) and compound (CP) perturbations.

The level 5 signatures of the CLUE ``beta`` release (``level5_beta_trt_{oe,xpr,cp}_n..x12328.gctx``: moderated z-scores of 12 328 genes per
signature) are read from disk in blocks (:class:`LincsLibrary`, HDF5 via ``h5py``, nothing is loaded into memory as a whole). The activity of a
regulator in a signature is its aREA enrichment score (:func:`pyrea.area_nes`, identical to ``viper``): the signatures are already differential, so
no further signature step is needed. Several networks are integrated like metaVIPER (:func:`pyrea.metaviper`); the networks (:class:`LincsNetworks`)
can be chosen per cell line or lineage, so analyses specific for a tumour type are possible.

Questions this answers for a candidate (e.g. ZEB1 or ZCCHC24):

- :func:`tf_response`: does the activity of the regulator fall when its own gene is knocked out (XPR) and rise when it is overexpressed (OE)?
- :func:`perturbation_targets`: which other regulators change when the gene is perturbed (top X)?
- :func:`perturbed_by`: for a regulator whose gene has no perturbation: the perturbation of which genes changes its activity most?
- :func:`tf_in_compounds`: which compounds raise or lower its activity, and which mechanisms of action are enriched?
"""

from __future__ import annotations

import hashlib
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, false_discovery_control, mannwhitneyu, norm, spearmanr

from .area_numpy import compact_regulons, metaviper, rank_transform, _nes_of

__all__ = ["LincsLibrary", "LincsNetworks", "RankedEffects", "TfResponse", "tf_response", "perturbation_targets", "perturbed_by",
           "perturbation_specificity", "tf_in_compounds", "compound_set_enrichment"]

_KINDS = {"trt_oe": "oe", "trt_xpr": "xpr", "trt_cp": "cp", "oe": "oe", "xpr": "xpr", "cp": "cp"}
_REQUIRED = ("sig_id", "cell_iname")
_SIG_COLUMNS = ("sig_id", "pert_type", "cell_iname", "cmap_name", "pert_iname", "pert_id", "pert_idose", "pert_itime", "pct_self_rank_q25", "tas",
                "cc_q75", "qc_pass", "is_hiq", "nsample")


def _read_table(x, usecols=None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if usecols is not None:                                   # the real siginfo has 1.2 million rows and 40 columns: read what is needed
        header = pd.read_csv(x, sep="\t", nrows=0).columns
        usecols = [c for c in header if c in usecols]
    return pd.read_csv(x, sep="\t", low_memory=False, usecols=usecols)


def _number(s) -> float:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m.group()) if m else np.nan


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


# ---------------------------------------------------------------------- the library of signatures
class LincsLibrary:
    """Signatures of the LINCS L1000 level 5 files, read from disk on demand.

    Parameters
    ----------
    siginfo : path or DataFrame
        ``siginfo_beta.txt`` (tab-separated): at least ``sig_id`` and ``cell_iname`` and the name of the perturbation (``pert_iname`` or
        ``cmap_name``: the gene symbol of an OE / XPR signature, the name of a compound); optional ``pert_type`` (signatures whose type does not fit the
        file, e.g. controls, are left out), ``pert_id``, ``pert_idose``, ``pert_itime`` (strings such as ``"10 uM"`` and ``"24 h"``, also read as
        numbers) and the quality columns ``qc_pass`` / ``is_hiq`` / ``pct_self_rank_q25`` / ``tas``. Only the columns named here are read from a file
        (the beta file has 40 columns and 1.2 million rows). Only signatures found in the given matrix files are kept.
    geneinfo : path or DataFrame
        ``geneinfo_beta.txt``: ``gene_id`` (the row ids of the matrices), ``gene_symbol``, optional ``feature_space``.
    oe, xpr, cp : path, optional
        The ``.gctx`` files (overexpression, CRISPR knockout, compound).
    cellinfo : path or DataFrame, optional
        ``cellinfo_beta.txt``: ``cell_iname``, ``cell_lineage``, ``primary_disease`` (for lineage selections and ``network_map``).
    compoundinfo : path or DataFrame, optional
        ``compoundinfo_beta.txt``: ``pert_id``, ``cmap_name``, ``target``, ``moa`` (annotation of compound results).

    Attributes
    ----------
    sig : DataFrame
        One row per signature (index ``sig_id``): ``kind`` (``"oe"``, ``"xpr"``, ``"cp"``), ``pert`` (gene or compound name), ``cell``,
        ``lineage``, ``disease``, ``dose``, ``time`` (numbers, if the columns exist), ``pert_id`` and all columns of ``siginfo``.
    genes : pd.Index
        Gene symbols of the rows of the matrices (duplicated symbols are dropped, the first is kept).
    """

    def __init__(self, siginfo, geneinfo, oe=None, xpr=None, cp=None, cellinfo=None, compoundinfo=None):
        self.files = {k: Path(v) for k, v in (("oe", oe), ("xpr", xpr), ("cp", cp)) if v is not None}
        if not self.files:
            raise ValueError("Give at least one of oe, xpr, cp (a .gctx file)")
        si = _read_table(siginfo, _SIG_COLUMNS)
        miss = [c for c in _REQUIRED if c not in si.columns]
        if "pert_iname" not in si.columns and "cmap_name" not in si.columns:
            miss.append("pert_iname or cmap_name")
        if miss:
            raise ValueError(f"siginfo lacks the columns {miss}; it has {list(si.columns)[:15]}")
        if "pert_iname" not in si.columns:
            si["pert_iname"] = si["cmap_name"]
        si = si.drop_duplicates("sig_id").set_index("sig_id")
        gi = _read_table(geneinfo)
        if not {"gene_id", "gene_symbol"} <= set(gi.columns):
            raise ValueError("geneinfo needs the columns gene_id and gene_symbol")
        sym = gi.assign(gene_id=gi["gene_id"].astype(str)).drop_duplicates("gene_id").set_index("gene_id")
        self._cols, parts = {}, []
        import h5py
        for kind, f in self.files.items():
            with h5py.File(f, "r") as h:
                ids = pd.Index(np.char.decode(h["/0/META/COL/id"][:].astype("S")))
                rows = pd.Index(np.char.decode(h["/0/META/ROW/id"][:].astype("S")))
                shape = h["/0/DATA/0/matrix"].shape
            if shape != (len(ids), len(rows)):
                raise ValueError(f"{f.name}: the matrix has the shape {shape}, expected (signatures, genes) = {(len(ids), len(rows))}")
            if kind == next(iter(self.files)):
                self._row_ids = rows
                self._row_perm = {kind: np.arange(len(rows))}
            elif rows.equals(self._row_ids):
                self._row_perm[kind] = np.arange(len(rows))
            elif rows.is_unique and set(rows) == set(self._row_ids):                 # the same genes in another order (the beta files differ)
                self._row_perm[kind] = rows.get_indexer(self._row_ids)
            else:
                raise ValueError(f"{f.name}: the genes (rows) differ from those of the other files")
            known = ids.isin(si.index)
            part = si.reindex(ids[known]).copy()
            if "pert_type" in part:                                   # drop e.g. control signatures stored in the file
                fits = part["pert_type"].astype(str).map(_KINDS).eq(kind).to_numpy()
                known[np.flatnonzero(known)[~fits]] = False
                part = part[fits]
            part["kind"] = kind
            part["file_row"] = np.flatnonzero(known)
            parts.append(part)
            self._cols[kind] = len(ids)
        sig = pd.concat(parts)
        sig["pert"] = sig["pert_iname"].astype(str)
        sig["cell"] = sig["cell_iname"].astype(str)
        sig["time"] = sig["pert_itime"].map(_number) if "pert_itime" in sig else np.nan
        sig["dose"] = sig["pert_idose"].map(_number) if "pert_idose" in sig else np.nan
        syms = sym["gene_symbol"].reindex(self._row_ids)
        if syms.isna().any():
            raise ValueError(f"{int(syms.isna().sum())} gene ids of the matrices are not in geneinfo (e.g. {list(self._row_ids[syms.isna().to_numpy()][:3])})")
        keep = ~syms.duplicated().to_numpy()
        self._row_keep = np.flatnonzero(keep)
        self.genes = pd.Index(syms.to_numpy()[keep].astype(str))
        self.gene_space = (sym["feature_space"].reindex(self._row_ids).to_numpy()[keep] if "feature_space" in sym else None)
        self.cellinfo = None
        sig["lineage"] = ""
        sig["disease"] = ""
        if cellinfo is not None:
            ci = _read_table(cellinfo).drop_duplicates("cell_iname").set_index("cell_iname")
            self.cellinfo = ci
            if "cell_lineage" in ci:
                sig["lineage"] = sig["cell"].map(ci["cell_lineage"]).fillna("").astype(str)
            if "primary_disease" in ci:
                sig["disease"] = sig["cell"].map(ci["primary_disease"]).fillna("").astype(str)
        self.compoundinfo = None
        if compoundinfo is not None:
            co = _read_table(compoundinfo).drop_duplicates("pert_id").set_index("pert_id")
            self.compoundinfo = co
        self.sig = sig
        if len(sig) == 0:
            raise ValueError("No signature of siginfo is in the matrix files (sig_id mismatch)")

    def __repr__(self) -> str:
        c = self.sig["kind"].value_counts().to_dict()
        return f"LincsLibrary({', '.join(f'{k}: {v} signatures' for k, v in c.items())}; {len(self.genes)} genes)"

    # ------------------------------------------------------------------ selecting
    def signatures(self, kind: Optional[str] = None, gene: Optional[str] = None, cell: "str | Sequence[str] | None" = None,
                   lineage: "str | Sequence[str] | None" = None, compound: Optional[str] = None, exclude_gene: Optional[str] = None,
                   times: Optional[Sequence[float]] = None, min_dose: Optional[float] = None, qc_pass: Optional[bool] = None) -> pd.DataFrame:
        """Signatures (rows of ``sig``) of a kind (``"oe"``, ``"xpr"``, ``"cp"``), perturbed gene or compound (name or ``pert_id``), cell lines
        or lineages; ``times`` (hours) and ``min_dose`` for compounds; ``exclude_gene`` leaves the signatures of a gene out; ``qc_pass=True`` keeps the
        signatures that passed the CLUE quality control (``qc_pass == 1``; no effect if siginfo has no such column)."""
        s = self.sig
        if qc_pass and "qc_pass" in s.columns:
            s = s[s["qc_pass"] == 1]
        if kind is not None:
            s = s[s["kind"] == _KINDS.get(kind, kind)]
        if gene is not None:
            s = s[s["pert"].str.upper() == str(gene).upper()]
        if exclude_gene is not None:
            s = s[s["pert"].str.upper() != str(exclude_gene).upper()]
        if compound is not None:
            s = s[(s["pert"].str.lower() == str(compound).lower()) | (s.get("pert_id", pd.Series("", index=s.index)).astype(str) == str(compound))]
        if cell is not None:
            s = s[s["cell"].isin([cell] if isinstance(cell, str) else list(cell))]
        if lineage is not None:
            s = s[s["lineage"].isin([lineage] if isinstance(lineage, str) else list(lineage))]
        if times is not None:
            s = s[s["time"].isin(list(times))]
        if min_dose is not None:
            s = s[s["dose"] >= min_dose]
        return s

    # ------------------------------------------------------------------ reading
    def _read_rows(self, kind: str, rows: np.ndarray, gap: int = 32) -> np.ndarray:
        import h5py
        order = np.argsort(rows, kind="stable")
        srt = rows[order]
        cols = self._row_perm[kind][self._row_keep]                            # columns of the file in the order of the genes
        out = np.empty((len(srt), len(cols)), dtype=np.float32)
        with h5py.File(self.files[kind], "r") as h:
            ds = h["/0/DATA/0/matrix"]
            i = 0
            while i < len(srt):
                j = i
                while j + 1 < len(srt) and srt[j + 1] - srt[j] <= gap:
                    j += 1
                block = ds[srt[i]:srt[j] + 1, :]
                out[i:j + 1] = block[srt[i:j + 1] - srt[i]][:, cols]
                i = j + 1
        res = np.empty_like(out)
        res[order] = out
        return res

    def read(self, sig_ids: Sequence[str]) -> pd.DataFrame:
        """Signatures as a DataFrame genes x ``sig_ids`` (float32)."""
        ids = list(sig_ids)
        if not ids:
            raise ValueError("No signatures given")
        missing = [i for i in ids if i not in self.sig.index]
        if missing:
            raise ValueError(f"Unknown signature id(s): {missing[:3]}")
        meta = self.sig.loc[ids]
        out = np.empty((len(ids), len(self.genes)), dtype=np.float32)
        for kind, g in meta.groupby("kind"):
            pos = meta.index.get_indexer(g.index)
            out[pos] = self._read_rows(kind, g["file_row"].to_numpy())
        return pd.DataFrame(out.T, index=self.genes, columns=ids)

    def blocks(self, sig_ids: Sequence[str], size: int = 2000):
        """Yield ``(ids, matrix)`` blocks (genes x signatures) of at most ``size`` signatures; the ids are read in file order within a kind."""
        meta = self.sig.loc[list(sig_ids)]
        for kind, g in meta.groupby("kind", sort=False):
            g = g.sort_values("file_row")
            for a in range(0, len(g), size):
                part = g.iloc[a:a + size]
                yield list(part.index), pd.DataFrame(self._read_rows(kind, part["file_row"].to_numpy()).T, index=self.genes, columns=list(part.index))

    def compound_info(self, pert_ids: Iterable[str]) -> pd.DataFrame:
        """``cmap_name``, ``target`` and ``moa`` of compounds (empty strings if ``compoundinfo`` was not given)."""
        ids = list(pert_ids)
        if self.compoundinfo is None:
            return pd.DataFrame({"cmap_name": "", "target": "", "moa": ""}, index=ids)
        co = self.compoundinfo.reindex(ids)
        return pd.DataFrame({c: co[c].fillna("").astype(str) if c in co else "" for c in ("cmap_name", "target", "moa")}, index=ids)

    # ------------------------------------------------------------------ activity
    def activity(self, sig_ids: Sequence[str], networks: "LincsNetworks", regulators: Optional[Sequence[str]] = None,
                 use: Optional[Sequence[str]] = None, network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0,
                 per_network: bool = False, size: int = 2000):
        """Activity (NES) of regulators in signatures: aREA of every network, integrated like metaVIPER.

        Parameters
        ----------
        sig_ids : sequence of str
        networks : LincsNetworks
        regulators : sequence of str, optional
            Default: all regulators of the networks (``use``).
        use : sequence of str, optional
            Names of the networks to use (default: all).
        network_map : dict, optional
            ``cell line or lineage -> network names``: a signature is integrated over the networks of its cell line (looked up first) or of its
            lineage only (**tumour-line-specific** activity); signatures of cells that are not in the map use all of ``use``.
        mvws : float
            Weight exponent of metaVIPER (``w = |NES| ** mvws``).
        per_network : bool
            Return a dict network -> DataFrame instead of the integrated DataFrame.

        Returns
        -------
        DataFrame regulators x signatures (the order of ``sig_ids``), or a dict.
        """
        ids = list(sig_ids)
        use = list(networks.names) if use is None else list(use)
        unknown = [n for n in use if n not in networks.regs]
        if unknown:
            raise ValueError(f"Unknown network(s) {unknown}; available: {networks.names}")
        meta = self.sig.loc[ids]
        sel = {n: {r: v for r, v in networks.regs[n].items() if regulators is None or r in set(map(str, regulators))} for n in use}
        regs = list(dict.fromkeys(r for n in use for r in sel[n]))
        if not regs:
            raise ValueError("None of the regulators is in the networks (with enough targets among the LINCS genes)")
        if regulators is not None:
            lost = sorted(set(map(str, regulators)) - set(regs))
            if lost:
                warnings.warn(f"Regulators without a usable regulon in the networks: {lost}")
        pos = {r: i for i, r in enumerate(regs)}
        allowed_all = None
        if network_map is not None:
            allowed_all = np.ones((len(use), len(ids)), dtype=bool)
            for j, (cell, lin) in enumerate(zip(meta["cell"], meta["lineage"])):
                names = network_map.get(cell, network_map.get(lin))
                if names is not None:
                    allowed_all[:, j] = [n in set(names) for n in use]
        out = np.full((len(regs), len(ids)), np.nan)
        per = {n: np.full((len(regs), len(ids)), np.nan) for n in use} if per_network else None
        col = {sid: j for j, sid in enumerate(ids)}
        for block_ids, X in self.blocks(ids, size):
            t1, t2 = rank_transform(X.to_numpy())
            arr = np.full((len(use), len(regs), len(block_ids)), np.nan)
            for k, n in enumerate(use):
                for r, comp in sel[n].items():
                    arr[k, pos[r], :] = _nes_of(t1, t2, *comp)
            js = np.array([col[b] for b in block_ids])
            allowed = None if allowed_all is None else allowed_all[:, js]
            out[:, js] = metaviper(arr, mvws=mvws, allowed=allowed)
            if per_network:
                for k, n in enumerate(use):
                    per[n][:, js] = arr[k]
        if per_network:
            return {n: pd.DataFrame(v, index=regs, columns=ids) for n, v in per.items()}
        return pd.DataFrame(out, index=regs, columns=ids)


# ---------------------------------------------------------------------- the networks
class LincsNetworks:
    """Regulons of several networks restricted to the genes of a :class:`LincsLibrary`.

    Parameters
    ----------
    library : LincsLibrary (or a sequence of gene symbols)
    networks : dict name -> path (``.rds``) or long-format regulon DataFrame, or a sequence of paths (named by the file stem)
        All networks are the default of every analysis; ``LincsLibrary.activity(use=...)`` and the ``networks=`` argument of the analyses select
        some (e.g. the network of a tumour type for the cell lines of that tumour type).
    minsize : int
        Minimum number of targets among the LINCS genes for a regulator to be used (VIPER's ``minsize``).
    regulators : sequence of str, optional
        Keep only these regulators (saves memory).
    cache_dir : path, optional
        The restricted regulons are cached per network (key: file, genes, ``minsize``, ``regulators``).
    regulon_reader : callable, optional
        ``reader(path) -> DataFrame`` instead of :func:`pyrea.read_regulon_rds`.

    Attributes
    ----------
    names : list of the network names; regs : dict network -> {regulator: (gene positions, mor, likelihood)}
    coverage : DataFrame, per network: regulators in the network, regulators usable (``minsize``) in LINCS.
    """

    def __init__(self, library, networks, minsize: int = 25, regulators: Optional[Sequence[str]] = None,
                 cache_dir: Union[str, Path, None] = None, regulon_reader=None):
        genes = list(library.genes if isinstance(library, LincsLibrary) else library)
        if isinstance(networks, Mapping):
            items = list(networks.items())
        else:
            items = [(Path(p).stem, p) for p in networks]
        if not items or len({n for n, _ in items}) != len(items):
            raise ValueError("Give networks with unique names (dict name -> path or DataFrame)")
        self.minsize, self.regs, rows = int(minsize), {}, []
        ghash = hashlib.sha1("|".join(genes).encode()).hexdigest()[:12]
        for name, item in items:
            f = None
            if cache_dir is not None and not isinstance(item, pd.DataFrame):
                st = Path(item).stat()
                key = hashlib.sha1(repr((str(Path(item).resolve()), st.st_size, st.st_mtime_ns, ghash, minsize,
                                         None if regulators is None else sorted(map(str, regulators)))).encode()).hexdigest()[:16]
                f = Path(cache_dir) / f"{''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(name))}-{key}.pkl"
                if f.exists():
                    with open(f, "rb") as fh:
                        self.regs[str(name)], n_all = pickle.load(fh)
                    rows.append({"network": str(name), "n_regulators": n_all, "n_usable": len(self.regs[str(name)])})
                    continue
            if isinstance(item, pd.DataFrame):
                reg = item
            elif regulon_reader is not None:
                reg = regulon_reader(item)
            else:
                from .rbackend.regulon import read_regulon_rds
                reg = read_regulon_rds(item)
            n_all = int(reg["source"].nunique())
            self.regs[str(name)] = compact_regulons(reg, genes, minsize, regulators)
            rows.append({"network": str(name), "n_regulators": n_all, "n_usable": len(self.regs[str(name)])})
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "wb") as fh:
                    pickle.dump((self.regs[str(name)], n_all), fh)
        self.names = [r["network"] for r in rows]
        self.coverage = pd.DataFrame(rows)
        self._sizes = {n: {r: len(v[0]) for r, v in d.items()} for n, d in self.regs.items()}

    def __repr__(self) -> str:
        return f"LincsNetworks({len(self.names)} network(s), minsize={self.minsize})"

    def regulators(self) -> list:
        """All regulators with a usable regulon in at least one network."""
        return sorted({r for d in self.regs.values() for r in d})

    def tf_coverage(self, tf: str) -> pd.DataFrame:
        """Per network: the number of targets of ``tf`` among the LINCS genes (NaN: not usable, no regulon or below ``minsize``)."""
        return pd.DataFrame({"network": self.names, "n_targets": [self._sizes[n].get(tf, np.nan) for n in self.names]})


# ---------------------------------------------------------------------- results
@dataclass
class RankedEffects:
    """A ranked table of effects (result of :func:`perturbation_targets`, :func:`perturbed_by`, :func:`tf_in_compounds`).

    ``table``: one row per regulator / perturbed gene / compound, with ``z`` (Stouffer z over cell lines of the activity change, standardised within
    cell line: positive = activity rises), ``z_median``, ``n_cells``, ``n_signatures``, ``frac_same_sign`` (share of cell lines with the sign of
    ``z``) and for ``perturbed_by`` / ``tf_in_compounds`` a normal-approximation ``p`` and ``q`` (the signatures are not fully independent: a
    ranking, not a proof). ``meta``: the settings. ``up(n)`` / ``down(n)``: the n strongest rises / falls.
    """

    table: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RankedEffects({len(self.table)} rows; {self.meta.get('what', '')})"

    def up(self, n: int = 25, **filters) -> pd.DataFrame:
        t = self._filter(**filters)
        return t.sort_values("z", ascending=False).head(n).reset_index(drop=True)

    def down(self, n: int = 25, **filters) -> pd.DataFrame:
        t = self._filter(**filters)
        return t.sort_values("z", ascending=True).head(n).reset_index(drop=True)

    def _filter(self, **filters) -> pd.DataFrame:
        t = self.table
        for k, v in filters.items():
            t = t[t[k] == v]
        return t


@dataclass
class TfResponse:
    """Result of :func:`tf_response`: ``table`` per signature (``nes``, ``expression_z`` of the gene itself, ``bg_percentile``,
    ``expected``), ``summary`` per kind, ``cells`` per kind and cell line (median activity and its percentile among the signatures of other
    genes in the same cell line), ``missing`` (kinds without signatures of the gene)."""

    table: pd.DataFrame
    summary: pd.DataFrame
    cells: pd.DataFrame
    missing: list = field(default_factory=list)


def _sample_background(lib: LincsLibrary, kind: str, cells: Iterable[str], exclude_gene: Optional[str], n: int, seed: int,
                       exclude_ids: Iterable[str] = (), qc_pass: bool = True) -> list:
    s = lib.signatures(kind, cell=sorted(set(cells)), exclude_gene=exclude_gene, qc_pass=qc_pass)
    s = s[~s.index.isin(set(exclude_ids))]
    if len(s) > n:
        s = s.sample(n, random_state=seed)
    return list(s.index)


# ---------------------------------------------------------------------- tf_response
def tf_response(lib: LincsLibrary, tf: str, networks: LincsNetworks, kinds: Sequence[str] = ("xpr", "oe"), cells: "str | Sequence[str] | None" = None,
                use: Optional[Sequence[str]] = None, network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0,
                background: int = 2000, seed: int = 0, qc_pass: bool = True) -> TfResponse:
    """Does the activity of a regulator fall when its gene is knocked out (XPR) and rise when it is overexpressed (OE)?

    The activity of ``tf`` is determined in the signatures of its own gene and compared with its activity in the signatures of **other genes in
    the same cell lines** (``background`` random signatures of the same kind): the question is not whether the activity changes but whether it
    changes more than by any perturbation. Expected direction: XPR negative, OE positive.

    Returns
    -------
    TfResponse
        ``table`` (per signature: ``nes``, ``expression_z`` of the gene itself, ``bg_percentile`` = share of the background in the same cell line
        with a lower activity, ``expected``: the activity moves in the expected direction), ``summary`` per kind (``n_signatures``, ``n_cells``,
        ``median_nes``, ``frac_expected`` of signatures and of cell lines, ``auc`` = probability that an own signature is more extreme in the
        expected direction than a background signature, ``z_cells`` / ``p_cells``: Stouffer over the cell lines of the percentile of the cell
        line's median activity among the background of that cell line, one-sided; ``expression_rho``: Spearman correlation of the gene's own level
        in the signature with the activity) and ``cells``.
    """
    rows, summ, cellrows, missing = [], [], [], []
    for kind in [_KINDS[k] for k in kinds]:
        own = lib.signatures(kind, gene=tf, cell=cells, qc_pass=qc_pass)
        if own.empty:
            missing.append(kind)
            continue
        expected = -1.0 if kind == "xpr" else 1.0
        act = lib.activity(list(own.index), networks, regulators=[tf], use=use, network_map=network_map, mvws=mvws).loc[tf]
        z = lib.read(list(own.index)).loc[tf] if tf in lib.genes else pd.Series(np.nan, index=own.index)
        bg_ids = _sample_background(lib, kind, own["cell"], tf, background, seed, qc_pass=qc_pass)
        if len(bg_ids) < 20:
            raise ValueError(f"Only {len(bg_ids)} background signatures of other genes in these cell lines")
        bg_act = lib.activity(bg_ids, networks, regulators=[tf], use=use, network_map=network_map, mvws=mvws).loc[tf]
        bg_cell = lib.sig.loc[bg_ids, "cell"]
        t = pd.DataFrame({"kind": kind, "cell": own["cell"], "time": own["time"], "nes": act, "expression_z": z.reindex(own.index)}, index=own.index)
        pooled = np.sort(bg_act.to_numpy())

        def pct(values, cell_names):
            res = []
            for v, c in zip(values, cell_names):
                b = bg_act[(bg_cell == c).to_numpy()].to_numpy()
                b = b if len(b) >= 20 else pooled
                res.append(float(np.mean(b < v) + 0.5 * np.mean(b == v)))
            return np.array(res)

        t["bg_percentile"] = pct(t["nes"], t["cell"])
        t["expected"] = np.sign(t["nes"]) == expected
        t.insert(0, "sig_id", t.index)
        rows.append(t)
        cm = t.groupby("cell")["nes"].median()
        cpct = pct(cm.to_numpy(), cm.index)
        zc = norm.ppf(np.clip(cpct, 1e-3, 1 - 1e-3)) * expected                         # in the expected direction: positive
        zs = zc.sum() / np.sqrt(len(zc))
        cellrows.append(pd.DataFrame({"kind": kind, "cell": cm.index, "n_signatures": t.groupby("cell").size().to_numpy(), "median_nes": cm.to_numpy(),
                                      "bg_percentile": cpct, "expected": np.sign(cm.to_numpy()) == expected}))
        own_dir = t["nes"].to_numpy() * expected
        bg_dir = bg_act.to_numpy() * expected
        u = mannwhitneyu(own_dir, bg_dir, alternative="greater")
        rho = spearmanr(t["expression_z"], t["nes"])[0] if t["expression_z"].notna().sum() >= 5 else np.nan
        summ.append({"kind": kind, "n_signatures": len(t), "n_cells": t["cell"].nunique(), "n_background": len(bg_ids), "median_nes": float(t["nes"].median()),
                     "frac_expected": float(t["expected"].mean()), "frac_cells_expected": float(np.mean(np.sign(cm.to_numpy()) == expected)),
                     "auc": float(u.statistic / (len(own_dir) * len(bg_dir))), "z_cells": float(zs), "p_cells": float(norm.sf(zs)),
                     "expression_rho": float(rho)})
    if not rows:
        raise ValueError(f"No signatures of {tf} for the kinds {list(kinds)}")
    return TfResponse(table=pd.concat(rows, ignore_index=True), summary=pd.DataFrame(summ), cells=pd.concat(cellrows, ignore_index=True), missing=missing)


# ---------------------------------------------------------------------- perturbation_targets
def perturbation_targets(lib: LincsLibrary, gene: str, networks: LincsNetworks, kind: str = "xpr", cells: "str | Sequence[str] | None" = None,
                         regulators: Optional[Sequence[str]] = None, use: Optional[Sequence[str]] = None,
                         network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0, background: int = 1500, min_cells: int = 1,
                         seed: int = 0, qc_pass: bool = True) -> RankedEffects:
    """Which regulators change their activity when ``gene`` is perturbed (``kind``: ``"xpr"`` knockout or ``"oe"`` overexpression)?

    The activity of all regulators of the networks (or ``regulators``) is determined in the signatures of the gene, each value is standardised
    against the activity of the same regulator in ``background`` signatures of **other genes in the same cell line** (``z`` per signature),
    the signatures of a cell line are summarised by their median and the cell lines by Stouffer's z. NES values of different regulators are not
    compared directly. Returns a :class:`RankedEffects` (``table`` with ``regulator``, ``z``, ``z_median``, ``n_cells``, ``n_signatures``, ``frac_same_sign``,
    ``self`` = the regulator of the perturbed gene itself; ``up()`` / ``down()``).
    """
    kind = _KINDS[kind]
    own = lib.signatures(kind, gene=gene, cell=cells, qc_pass=qc_pass)
    if own.empty:
        raise ValueError(f"No {kind} signatures of {gene}")
    act = lib.activity(list(own.index), networks, regulators=regulators, use=use, network_map=network_map, mvws=mvws)
    bg_ids = _sample_background(lib, kind, own["cell"], gene, background, seed, qc_pass=qc_pass)
    if len(bg_ids) < 20:
        raise ValueError(f"Only {len(bg_ids)} background signatures of other genes in these cell lines")
    bg = lib.activity(bg_ids, networks, regulators=list(act.index), use=use, network_map=network_map, mvws=mvws)
    bg_cell = lib.sig.loc[bg_ids, "cell"].to_numpy()
    pooled_m, pooled_s = bg.mean(axis=1), bg.std(axis=1).replace(0, np.nan)
    zsig = pd.DataFrame(index=act.index, columns=act.columns, dtype=float)
    for cell, g in own.groupby("cell"):
        sel = bg.loc[:, bg_cell == cell]
        m, s = (sel.mean(axis=1), sel.std(axis=1).replace(0, np.nan)) if sel.shape[1] >= 20 else (pooled_m, pooled_s)
        zsig[list(g.index)] = (act[list(g.index)].sub(m, axis=0)).div(s, axis=0)
    percell = zsig.T.groupby(own["cell"].to_numpy()).median().T
    k = percell.notna().sum(axis=1)
    zsum = percell.sum(axis=1, skipna=True) / np.sqrt(k.replace(0, np.nan))
    sign = np.sign(zsum.to_numpy())[:, None]
    frac = ((np.sign(percell.to_numpy()) == sign) & percell.notna().to_numpy()).sum(axis=1) / k.replace(0, np.nan).to_numpy()
    tab = pd.DataFrame({"regulator": act.index, "z": zsum.to_numpy(), "z_median": percell.median(axis=1).to_numpy(), "n_cells": k.to_numpy(),
                        "n_signatures": len(own), "frac_same_sign": frac, "self": act.index.str.upper() == str(gene).upper()})
    tab = tab[tab["n_cells"] >= min_cells].reset_index(drop=True)
    return RankedEffects(tab, {"what": f"regulators changed by {kind} of {gene}", "gene": gene, "kind": kind, "n_signatures": len(own),
                               "n_background": len(bg_ids), "cells": sorted(own["cell"].unique())})


# ---------------------------------------------------------------------- scans over many signatures
def _scan(lib: LincsLibrary, tf: str, sig_ids: Sequence[str], networks: LincsNetworks, use, network_map, mvws, size, cache_dir) -> pd.Series:
    f = None
    if cache_dir is not None:
        key = hashlib.sha1(repr((tf, hashlib.sha1("|".join(sig_ids).encode()).hexdigest(), sorted(networks.names if use is None else use),
                                 networks.minsize, mvws, None if network_map is None else sorted((k, tuple(v)) for k, v in network_map.items()),
                                 sorted((str(k), str(lib.files[k].stat().st_mtime_ns)) for k in set(lib.sig.loc[sig_ids, "kind"])))).encode()).hexdigest()[:20]
        f = Path(cache_dir) / f"scan-{tf}-{key}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                return pickle.load(fh)
    act = lib.activity(sig_ids, networks, regulators=[tf], use=use, network_map=network_map, mvws=mvws, size=size).loc[tf]
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(act, fh)
    return act


def _aggregate(df: pd.DataFrame, group: str, cell: str, min_cells: int) -> pd.DataFrame:
    """z per signature -> median per group and cell -> Stouffer over the cells of a group."""
    pc = df.groupby([group, cell])["z"].median().rename("z_cell").reset_index()
    g = pc.groupby(group)["z_cell"]
    out = pd.DataFrame({"z": g.sum() / np.sqrt(g.size()), "z_median": g.median(), "n_cells": g.size()})
    same = np.sign(pc["z_cell"].to_numpy()) == np.sign(out["z"]).reindex(pc[group]).to_numpy()
    out["frac_same_sign"] = pd.Series(same, index=pc.index).groupby(pc[group].to_numpy()).mean().reindex(out.index)
    out = out.join(df.groupby(group).size().rename("n_signatures"))
    out = out[out["n_cells"] >= min_cells]
    out["p"] = 2 * norm.sf(np.abs(out["z"]))
    return out.reset_index()


def perturbed_by(lib: LincsLibrary, tf: str, networks: LincsNetworks, kinds: Sequence[str] = ("xpr", "oe"), cells: "str | Sequence[str] | None" = None,
                 use: Optional[Sequence[str]] = None, network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0,
                 min_cells: int = 2, size: int = 4000, cache_dir: Union[str, Path, None] = None, qc_pass: bool = True) -> RankedEffects:
    """The perturbation of which genes changes the activity of ``tf`` most (for regulators whose own gene has no perturbation)?

    The activity of ``tf`` is determined in **all** signatures of the kinds (``"xpr"`` knockout, ``"oe"`` overexpression; chunked, cached with
    ``cache_dir``), standardised within the cell line and kind (so a cell line with a high baseline does not dominate), summarised per perturbed gene
    by the median per cell line and Stouffer's z over cell lines (``min_cells`` cell lines are required). Returns a :class:`RankedEffects`:
    ``table`` with ``kind``, ``gene``, ``z`` (negative: the activity falls), ``z_median``, ``n_cells``, ``n_signatures``, ``frac_same_sign``, ``p`` and ``q``
    (normal approximation, Benjamini-Hochberg within a kind; replicate signatures are correlated, read ``q`` as a ranking aid).
    """
    parts = []
    for kind in [_KINDS[k] for k in kinds]:
        s = lib.signatures(kind, cell=cells, qc_pass=qc_pass)
        if s.empty:
            continue
        act = _scan(lib, tf, list(s.index), networks, use, network_map, mvws, size, cache_dir)
        d = pd.DataFrame({"kind": kind, "gene": s["pert"], "cell": s["cell"], "nes": act.reindex(s.index)})
        d = d.dropna(subset=["nes"])
        grp = d.groupby(["kind", "cell"])["nes"]
        sd = grp.transform("std").replace(0, np.nan)
        d["z"] = (d["nes"] - grp.transform("mean")) / sd
        agg = _aggregate(d.dropna(subset=["z"]), "gene", "cell", min_cells)
        agg.insert(0, "kind", kind)
        agg["q"] = _bh(agg["p"].to_numpy())
        agg["self"] = agg["gene"].str.upper() == str(tf).upper()
        parts.append(agg)
    if not parts:
        raise ValueError(f"No signatures of the kinds {list(kinds)}")
    return RankedEffects(pd.concat(parts, ignore_index=True), {"what": f"genes whose perturbation changes {tf}", "tf": tf, "kinds": [_KINDS[k] for k in kinds]})


def perturbation_specificity(lib: LincsLibrary, tf: str, genes: Sequence[str], networks: LincsNetworks, kind: str = "xpr",
                             cells: "str | Sequence[str] | None" = None, use: Optional[Sequence[str]] = None,
                             network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0, background: int = 1000,
                             seed: int = 0, qc_pass: bool = True, z_cut: float = 3.0) -> pd.DataFrame:
    """Is the change of ``tf`` after the perturbation of a gene specific, or does that perturbation change many regulators?

    A knockout of an essential gene (e.g. of a transcription or replication factor) shifts the state of the cell and with it the activity of hundreds
    of regulators; a hit of :func:`perturbed_by` among them says little about ``tf``. For every gene of ``genes`` (e.g. the top of :func:`perturbed_by`) the
    activity of **all** regulators is determined with :func:`perturbation_targets` and ``tf`` is placed among them: ``z`` of ``tf``, ``rank`` by ``|z|``
    among all regulators (1 = the regulator that changes most), ``percentile`` (rank / regulators; small = specific) and ``n_changed`` (regulators with
    ``|z| >= z_cut``: a large number means a broad response). Returns a DataFrame with one row per gene.
    """
    rows = []
    for g in genes:
        try:
            e = perturbation_targets(lib, g, networks, kind=kind, cells=cells, use=use, network_map=network_map, mvws=mvws,
                                     background=background, seed=seed, qc_pass=qc_pass)
        except ValueError:
            rows.append({"gene": g, "n_signatures": 0})
            continue
        t = e.table.dropna(subset=["z"])
        me = t[t["regulator"] == tf]
        if me.empty:
            rows.append({"gene": g, "n_signatures": e.meta["n_signatures"]})
            continue
        z = float(me["z"].iloc[0])
        rank = int(1 + (t["z"].abs() > abs(z)).sum())
        rows.append({"gene": g, "n_signatures": e.meta["n_signatures"], "n_cells": int(me["n_cells"].iloc[0]), "z": z, "rank": rank,
                     "n_regulators": len(t), "percentile": rank / len(t), "n_changed": int((t["z"].abs() >= z_cut).sum()),
                     "n_up": int((t["z"] >= z_cut).sum()), "n_down": int((t["z"] <= -z_cut).sum())})
    return pd.DataFrame(rows, columns=["gene", "n_signatures", "n_cells", "z", "rank", "n_regulators", "percentile", "n_changed", "n_up", "n_down"])


# ---------------------------------------------------------------------- compounds
def tf_in_compounds(lib: LincsLibrary, tf: str, networks: LincsNetworks, cells: "str | Sequence[str] | None" = None, lineage: "str | Sequence[str] | None" = None,
                    times: Optional[Sequence[float]] = None, min_dose: Optional[float] = None, use: Optional[Sequence[str]] = None,
                    network_map: Optional[Mapping[str, Sequence[str]]] = None, mvws: float = 1.0, min_cells: int = 2, size: int = 4000,
                    cache_dir: Union[str, Path, None] = None, qc_pass: bool = True) -> RankedEffects:
    """Which compounds raise or lower the activity of ``tf`` (CP signatures)?

    Selection by ``cells`` / ``lineage`` / ``times`` (hours) / ``min_dose``. The activity is standardised within cell line and time (dose and plate
    effects are not modelled), summarised per compound by the median per cell line and Stouffer's z over cell lines. ``table``: ``pert_id``,
    ``compound`` (``cmap_name``), ``moa``, ``target``, ``z``, ``z_median``, ``n_cells``, ``n_signatures``, ``frac_same_sign``, ``p``, ``q``. See
    :func:`compound_set_enrichment` for the mechanisms of action among the top compounds.
    """
    s = lib.signatures("cp", cell=cells, lineage=lineage, times=times, min_dose=min_dose, qc_pass=qc_pass)
    if s.empty:
        raise ValueError("No compound signatures for this selection")
    act = _scan(lib, tf, list(s.index), networks, use, network_map, mvws, size, cache_dir)
    pid = s["pert_id"].astype(str) if "pert_id" in s else s["pert"]
    d = pd.DataFrame({"pert_id": pid, "cell": s["cell"], "time": s["time"], "nes": act.reindex(s.index)}).dropna(subset=["nes"])
    grp = d.groupby(["cell", "time"], dropna=False)["nes"]
    d["z"] = (d["nes"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    agg = _aggregate(d.dropna(subset=["z"]), "pert_id", "cell", min_cells)
    info = lib.compound_info(agg["pert_id"])
    agg.insert(1, "compound", info["cmap_name"].to_numpy())
    agg["moa"], agg["target"] = info["moa"].to_numpy(), info["target"].to_numpy()
    agg["q"] = _bh(agg["p"].to_numpy())
    return RankedEffects(agg, {"what": f"compounds changing {tf}", "tf": tf, "n_signatures": len(s)})


def compound_set_enrichment(effects: RankedEffects, top: int = 100, direction: str = "down", column: str = "moa", min_size: int = 3) -> pd.DataFrame:
    """Which mechanisms of action (``column="moa"``) or targets (``"target"``) are enriched among the ``top`` compounds that lower (``direction="down"``)
    or raise (``"up"``) the activity? One-sided Fisher test against all compounds of the table, Benjamini-Hochberg. Terms are split at ``|``."""
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    t = effects.table
    if column not in t.columns:
        raise ValueError(f"The table has no column {column!r}")
    order = t.sort_values("z", ascending=(direction == "down"))
    hit = set(order.head(top).index)
    terms = t[column].fillna("").astype(str).str.split("|")
    allterms = {x.strip() for lst in terms for x in lst if x.strip() and x.strip() != '""'}
    n, k = len(t), len(hit)
    rows = []
    for term in sorted(allterms):
        has = terms.map(lambda l: term in {x.strip() for x in l})
        a = int(has.loc[list(hit)].sum())
        tot = int(has.sum())
        if tot < min_size or a == 0:
            continue
        _, p = fisher_exact([[a, k - a], [tot - a, n - k - (tot - a)]], alternative="greater")
        rows.append({"term": term, "n_top": a, "n_total": tot, "fold": (a / k) / (tot / n), "p": p})
    out = pd.DataFrame(rows, columns=["term", "n_top", "n_total", "fold", "p"])
    if out.empty:
        return out.assign(q=[])
    out["q"] = _bh(out["p"].to_numpy())
    return out.sort_values("p", ignore_index=True)

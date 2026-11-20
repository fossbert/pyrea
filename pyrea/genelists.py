"""Collections of named gene lists with provenance, and over-representation analysis against gene sets.

:class:`GeneListCollection` gathers gene lists (e.g. the shared targets of regulator pairs from
:func:`pyrea.regulon_overlap`) over several analyses, keeps for each list where it came from, and exports them to Excel,
CSV or GMT. :func:`gene_set_enrichment` tests a gene list for over-representation of the terms of gene set collections
(the bundled ones, see :func:`pyrea.load_genesets`, or any dict of name -> genes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, hypergeom

__all__ = ["GeneListCollection", "gene_set_enrichment", "available_genesets", "resolve_genesets"]


def available_genesets() -> pd.DataFrame:
    """The gene set collections bundled with pyrea (MSigDB v7.4): one row per collection and species.

    Columns: ``collection`` (the name to pass to :func:`gene_set_enrichment`, ``RegulonOverlap.enrich`` or
    :func:`pyrea.load_genesets`), ``description``, ``species``, ``n_sets`` and the number of genes per set
    (``size_min``, ``size_median``, ``size_max``).
    """
    from .utils import GENESET_COLLECTIONS, GENESET_SPECIES, load_genesets
    desc = {"h": "MSigDB Hallmark", "pid": "C2 canonical pathways: PID", "wikipathways": "C2 canonical pathways: WikiPathways",
            "reactome": "C2 canonical pathways: Reactome"}
    rows = []
    for c in GENESET_COLLECTIONS:
        for sp in GENESET_SPECIES:
            sizes = np.array([len(v) for v in load_genesets(c, sp, verbose=False).values()])
            rows.append({"collection": c, "description": desc[c], "species": sp, "n_sets": len(sizes), "size_min": int(sizes.min()),
                         "size_median": int(np.median(sizes)), "size_max": int(sizes.max())})
    return pd.DataFrame(rows)


def resolve_genesets(genesets, species: str = "human"):
    """Gene sets as a dict term -> genes, from a dict or from the name(s) of bundled collections.

    Parameters
    ----------
    genesets : mapping, str or sequence of str
        A mapping term -> genes is returned as it is (second return value ``None``); a collection name (``"h"``, ``"reactome"``,
        ``"pid"``, ``"wikipathways"``; see :func:`available_genesets`) or a list of names loads and combines the bundled collections.
    species : {"human", "mouse"}

    Returns
    -------
    (dict, dict or None)
        The gene sets and, for bundled collections, the collection of every term.
    """
    from .utils import _GENESET_ALIASES, GENESET_COLLECTIONS, load_genesets
    if isinstance(genesets, Mapping):
        return dict(genesets), None
    names = [genesets] if isinstance(genesets, str) else list(genesets)
    if not names or not all(isinstance(n, str) for n in names):
        raise ValueError("genesets must be a mapping term -> genes or the name(s) of bundled collections: "
                         + ", ".join(GENESET_COLLECTIONS))
    sets, coll = {}, {}
    for n in names:
        c = _GENESET_ALIASES.get(n.lower(), n.lower())
        for term, genes in load_genesets(c, species, verbose=False).items():
            key = term if term not in sets else f"{c}:{term}"
            sets[key], coll[key] = genes, c
    return sets, coll


class GeneListCollection:
    """Named gene lists with metadata, for collecting results over analyses and exporting them later.

    Examples
    --------
    ::

        col = pr.GeneListCollection()
        ov.collect(col, label="STAD TCGA")                     # the shared targets of every regulator pair
        col.add("EMT core", ["VIM", "CDH2", "FN1"], source="manual")
        col.summary()                                          # one row per list: size and provenance
        col.export("shared_targets.xlsx")                      # sheets: lists, genes (long), plus extra tables
        col.to_gmt("shared_targets.gmt")                       # for GSEA / other tools
    """

    def __init__(self):
        self._sets: dict = {}
        self._meta: dict = {}

    # ------------------------------------------------------------------ container
    def __len__(self) -> int:
        return len(self._sets)

    def __contains__(self, name: str) -> bool:
        return name in self._sets

    def __iter__(self):
        return iter(self._sets)

    def __getitem__(self, name: str) -> list:
        return list(self._sets[name])

    def __repr__(self) -> str:
        n = sum(len(v) for v in self._sets.values())
        return f"GeneListCollection({len(self._sets)} lists, {n} gene entries)"

    @property
    def names(self) -> list:
        return list(self._sets)

    def add(self, name: str, genes: Iterable[str], overwrite: bool = False, **meta) -> "GeneListCollection":
        """Add a list (duplicates removed, sorted); ``meta`` is stored with it (shown in :meth:`summary`). Returns the collection."""
        if name in self._sets and not overwrite:
            raise ValueError(f"A list named {name!r} exists; use overwrite=True or another name")
        self._sets[name] = sorted(set(map(str, genes)))
        self._meta[name] = dict(meta)
        return self

    def remove(self, name: str) -> "GeneListCollection":
        del self._sets[name], self._meta[name]
        return self

    def meta(self, name: str) -> dict:
        return dict(self._meta[name])

    def to_dict(self) -> dict:
        """Name -> sorted list of genes."""
        return {k: list(v) for k, v in self._sets.items()}

    def union(self, names: Optional[Sequence[str]] = None) -> list:
        """Genes in at least one of the lists (default: all)."""
        names = self.names if names is None else list(names)
        return sorted(set().union(*[self._sets[n] for n in names])) if names else []

    def intersection(self, names: Sequence[str]) -> list:
        """Genes in all of the given lists."""
        names = list(names)
        if not names:
            return []
        return sorted(set.intersection(*[set(self._sets[n]) for n in names]))

    def add_overlap(self, ov, label: Optional[str] = None, signs: Optional[tuple] = None, overwrite: bool = False) -> "GeneListCollection":
        """Add the shared-target lists of a :class:`pyrea.RegulonOverlap` (see ``RegulonOverlap.gene_sets``); ``label`` prefixes the names."""
        info = ov._gene_set_info(signs)
        for name, rec in info.items():
            full = f"{label}: {name}" if label else name
            self.add(full, rec.pop("genes"), overwrite=overwrite, source=label or "", **rec)
        return self

    # ------------------------------------------------------------------ tables
    def summary(self) -> pd.DataFrame:
        """One row per list: ``name``, ``n_genes`` and the metadata (one column per key)."""
        rows = [{"name": n, "n_genes": len(g), **self._meta[n]} for n, g in self._sets.items()]
        return pd.DataFrame(rows, columns=["name", "n_genes"] + sorted({k for r in rows for k in r if k not in ("name", "n_genes")}))

    def to_frame(self) -> pd.DataFrame:
        """Long format: one row per list and gene (``set``, ``gene``)."""
        return pd.DataFrame([(n, g) for n, genes in self._sets.items() for g in genes], columns=["set", "gene"])

    def membership(self) -> pd.DataFrame:
        """Genes x lists, True where the gene is in the list."""
        genes = self.union()
        return pd.DataFrame({n: pd.Series(genes).isin(v).to_numpy() for n, v in self._sets.items()}, index=genes)

    # ------------------------------------------------------------------ export
    def to_gmt(self, path: Union[str, Path], description: Optional[Mapping[str, str]] = None) -> Path:
        """Write the lists as a GMT file (name, description, genes; tab separated). Returns the path."""
        path = Path(path)
        with open(path, "w") as fh:
            for n, genes in self._sets.items():
                desc = (description or {}).get(n, f"{len(genes)} genes")
                fh.write("\t".join([n, desc] + list(genes)) + "\n")
        return path

    def export(self, path: Union[str, Path], extra: Optional[Mapping[str, pd.DataFrame]] = None) -> Path:
        """Export to Excel (``.xlsx``: sheets ``lists``, ``genes`` and one per ``extra`` table), GMT (``.gmt``) or a folder of CSV files.

        Parameters
        ----------
        path : str or Path
            ``*.xlsx`` for one workbook, ``*.gmt`` for gene sets only, anything else is treated as a directory for
            ``lists.csv``, ``genes.csv`` and ``<name>.csv`` per extra table.
        extra : mapping name -> DataFrame, optional
            Further tables (sheet / file names; Excel sheet names are cut to 31 characters).
        """
        path = Path(path)
        tables = {"lists": self.summary(), "genes": self.to_frame(), **{k: v for k, v in (extra or {}).items()}}
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for name, df in tables.items():
                    df.to_excel(xw, sheet_name=str(name)[:31], index=not isinstance(df.index, pd.RangeIndex))
        elif path.suffix.lower() == ".gmt":
            self.to_gmt(path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(path / f"{name}.csv", index=not isinstance(df.index, pd.RangeIndex))
        return path


def gene_set_enrichment(genes: Iterable[str], genesets, universe: Optional[Iterable[str]] = None,
                        min_size: int = 5, max_size: int = 500, min_overlap: int = 2, species: str = "human") -> pd.DataFrame:
    """Over-representation of gene sets in a gene list (one-sided hypergeometric test).

    Parameters
    ----------
    genes : iterable of str
        The gene list (e.g. shared targets); genes outside ``universe`` are ignored.
    genesets : mapping term -> genes, str or sequence of str
        A mapping, or just the **name of a bundled collection**: ``"h"`` (Hallmark), ``"reactome"``, ``"pid"``, ``"wikipathways"``, or
        a list of them (``["h", "reactome"]``; see :func:`available_genesets`). The result then has a ``collection`` column.
    universe : iterable of str, optional
        All genes that could be in the list (default: the union of all gene set members). The gene sets are restricted to it.
        A universe that matches the genes that could have been found (e.g. the genes of the regulon network or the expressed
        genes) matters: with the default, terms are tested against a universe made of annotated genes only.
    min_size, max_size : int
        Terms with fewer / more members in the universe are not tested.
    min_overlap : int
        Terms with fewer genes in the list are not reported (they are still counted as tests for the correction).
    species : {"human", "mouse"}
        Species of the bundled collections (ignored for a mapping).

    Returns
    -------
    pd.DataFrame
        One row per reported term, sorted by p: ``term``, ``n_genes`` (list size in the universe), ``n_term``, ``overlap``,
        ``expected``, ``fold``, ``p``, ``q`` (Benjamini-Hochberg over all tested terms; with several collections over all of them
        together) and ``genes`` (the overlapping genes, ``;`` separated); with bundled collections also ``collection`` (second column).
    """
    resolved, coll = resolve_genesets(genesets, species)
    sets = {k: set(map(str, v)) for k, v in resolved.items()}
    uni = set(map(str, universe)) if universe is not None else set().union(*sets.values())
    g = set(map(str, genes)) & uni
    k, n = len(g), len(uni)
    rows = []
    for term, members in sets.items():
        m = members & uni
        if not (min_size <= len(m) <= max_size):
            continue
        o = g & m
        exp = k * len(m) / n if n else np.nan
        rows.append({"term": term, "n_genes": k, "n_term": len(m), "overlap": len(o), "expected": exp,
                     "fold": len(o) / exp if exp else np.nan,
                     "p": float(hypergeom.sf(len(o) - 1, n, len(m), k)) if k else np.nan, "genes": ";".join(sorted(o))})
    out = pd.DataFrame(rows, columns=["term", "n_genes", "n_term", "overlap", "expected", "fold", "p", "genes"])
    if len(out) and out["p"].notna().all():
        out["q"] = false_discovery_control(out["p"].to_numpy(), method="bh") if len(out) > 1 else out["p"]
    else:
        out["q"] = np.nan
    out = out[["term", "n_genes", "n_term", "overlap", "expected", "fold", "p", "q", "genes"]]
    if coll is not None:
        out.insert(1, "collection", out["term"].map(coll))
    return out[out["overlap"] >= min_overlap].sort_values("p").reset_index(drop=True)

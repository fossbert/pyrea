"""Gene list scores and background variables (covariates) for samples.

- :func:`score_genesets`: scores gene lists per sample with aREA (PAM50, EMT, hypoxia, any bundled collection, ...), the same machinery as the
  protein activity: a gene list is a regulon with all modes of regulation 1, the signature of a sample is taken relative to the cohort
  (``method="mad"``), the result is the normalised enrichment score of the list in that signature (needs R).
- A small registry of **background variables** for samples, stored like the gene sets: :func:`available_covariates`,
  :func:`load_covariates`, :func:`add_covariates`, :func:`remove_covariates`. The tables bundled with the package (``pyrea/data/covariates``) and
  the tables you add (directory ``$PYREA_DATA/covariates``, default ``~/.pyrea/covariates``) are used alike. Samples are identified by their name
  (for TCGA the 15-character ``TCGA-XX-XXXX-01``, see :func:`pyrea.tcga_sample_id`).
- :func:`resolve_covariates`: turns the several ways to give covariates (registry names, a table, gene lists to score) into one table for a cohort;
  used by :func:`pyrea.activity_specificity_across_cohorts`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["score_genesets", "available_covariates", "load_covariates", "add_covariates", "remove_covariates", "resolve_covariates"]

_BUNDLED = Path(__file__).resolve().parent / "data" / "covariates"


# ---------------------------------------------------------------------- gene list scores
def _as_sets(genesets, species: str) -> tuple:
    """(dict name -> genes, collection map or None); a bare list of genes becomes {"score": genes}."""
    from .genelists import resolve_genesets
    if isinstance(genesets, Mapping):
        return {str(k): list(map(str, v)) for k, v in genesets.items()}, None
    if isinstance(genesets, str):
        return resolve_genesets(genesets, species)
    if isinstance(genesets, (list, tuple, set, np.ndarray, pd.Index, pd.Series)):
        items = list(genesets)
        if items and all(isinstance(g, str) for g in items):
            names = {str(g).lower() for g in items}
            from .utils import GENESET_COLLECTIONS, _GENESET_ALIASES
            if names <= set(GENESET_COLLECTIONS) | set(_GENESET_ALIASES):
                return resolve_genesets(items, species)                              # a list of collection names
            return {"score": [str(g) for g in items]}, None
    raise ValueError("genesets must be a list of genes, a dict name -> genes or the name(s) of a bundled collection")


def score_genesets(expression: pd.DataFrame, genesets, method: str = "mad", min_genes: int = 5, species: str = "human",
                   nes: bool = True) -> pd.DataFrame:
    """Score gene lists in every sample with aREA (needs R).

    Parameters
    ----------
    expression : pd.DataFrame
        Genes x samples, normalised (log) expression of **one cohort**.
    genesets : list of str, dict, or str
        One gene list (the column is called ``"score"``), a dict ``name -> genes`` (PAM50, EMT, hypoxia, ...), or the name of a bundled
        collection (``"h"``, ``"reactome"``, ...; see :func:`pyrea.available_genesets`) or a list of them.
    method : {"mad", "scale", "rank", "none", "ttest"}
        Signature of a sample (as in :func:`pyrea.viper_activity`). ``"mad"``: relative to the median and MAD of every gene across the
        samples of the cohort, so the scores say *higher or lower than in the rest of this cohort* and are not comparable between cohorts.
    min_genes : int
        Lists with fewer genes present in ``expression`` are left out (with a warning).
    species : {"human", "mouse"}
        For bundled collections.
    nes : bool
        Normalised enrichment score (default) or the raw enrichment.

    Returns
    -------
    pd.DataFrame
        Samples x gene lists. ``df.attrs["n_genes"]`` holds the number of genes of every list that were found in ``expression``.

    Notes
    -----
    A gene list is a regulon with mode of regulation 1 for all genes, and the score is the aREA enrichment of the list in the signature
    of the sample, ranked over **all** genes of ``expression`` (``eset_filter=False``). The sign follows the direction of the genes (up in the
    sample: positive). For signatures where genes go in both directions give two lists. A list is not a subtype classifier: PAM50 genes scored
    this way give a signature score, not the PAM50 subtype.

    Examples
    --------
    ::

        scores = pr.score_genesets(expression, {"EMT": emt_genes, "Hypoxia": hypoxia_genes})
        hallmarks = pr.score_genesets(expression, "h")
    """
    from .rbackend.viper import viper_activity
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("expression needs unique gene names (rows) and sample names (columns)")
    sets, coll = _as_sets(genesets, species)
    genes = set(map(str, expression.index))
    rows, n_found, dropped = [], {}, []
    for name, members in sets.items():
        present = list(dict.fromkeys(g for g in members if g in genes))
        if len(present) < min_genes:
            dropped.append((name, len(present)))
            continue
        n_found[name] = len(present)
        rows.append(pd.DataFrame({"source": name, "target": present, "mor": 1.0, "likelihood": 1.0}))
    if dropped:
        import warnings
        warnings.warn(f"{len(dropped)} gene list(s) left out with fewer than {min_genes} genes in the expression: "
                      + ", ".join(f"{n} ({k})" for n, k in dropped[:6]) + (" ..." if len(dropped) > 6 else ""))
    if not rows:
        raise ValueError(f"None of the gene lists has {min_genes} or more genes in the expression")
    expr = expression.copy()
    expr.index = expr.index.astype(str)
    act = viper_activity(expr, pd.concat(rows, ignore_index=True), method=method, minsize=min_genes, eset_filter=False, nes=nes)
    out = act.T.reindex(index=expression.columns, columns=list(n_found))              # samples in the order of the expression, lists as given
    out.attrs["n_genes"] = pd.Series(n_found).reindex(out.columns)
    return out


# ---------------------------------------------------------------------- the registry of background variables
def _user_dir() -> Path:
    return Path(os.environ.get("PYREA_DATA") or (Path.home() / ".pyrea")) / "covariates"


def _read_index(directory: Path) -> dict:
    f = directory / "index.json"
    if not f.exists():
        return {"tables": {}}
    with open(f) as fh:
        return json.load(fh)


def _tables() -> dict:
    """name -> (directory, metadata) of the bundled and the user tables."""
    out = {}
    for loc, d in (("bundled", _BUNDLED), ("user", _user_dir())):
        for name, meta in _read_index(d).get("tables", {}).items():
            out[name] = (d, loc, meta)
    return out


def _variables() -> dict:
    """variable -> (table name, directory, location, table meta)."""
    out = {}
    for name, (d, loc, meta) in _tables().items():
        for var in meta["columns"]:
            out[var] = (name, d, loc, meta)
    return out


def _read_table(directory: Path, name: str, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(directory / meta["file"], index_col=0, dtype={0: str})
    df.index = df.index.astype(str)
    return df


def available_covariates() -> pd.DataFrame:
    """The background variables of the registry (bundled and added by you): ``variable``, ``description``, ``table``, ``source``,
    ``location`` (``"bundled"`` / ``"user"``) and ``n_samples`` with a value."""
    rows = []
    for var, (name, d, loc, meta) in _variables().items():
        tab = _read_table(d, name, meta)
        rows.append({"variable": var, "description": meta["columns"][var], "table": name, "source": meta.get("source", ""),
                     "location": loc, "n_samples": int(tab[var].notna().sum())})
    return pd.DataFrame(rows, columns=["variable", "description", "table", "source", "location", "n_samples"])


def load_covariates(names: "str | Sequence[str] | None" = None, samples: Optional[Iterable[str]] = None,
                    sample_id: Optional[Callable[[str], str]] = None) -> pd.DataFrame:
    """Background variables of samples from the registry.

    Parameters
    ----------
    names : str or sequence of str, optional
        Variables (see :func:`available_covariates`); default: all.
    samples : iterable of str, optional
        Sample names to return rows for (in this order; NaN where a sample has no value). Without it, the table of the variables with the
        registry's sample keys (TCGA: ``TCGA-XX-XXXX-01``) is returned.
    sample_id : callable, optional
        Maps ``samples`` to the registry's keys (e.g. :func:`pyrea.tcga_sample_id` for ``"TCGA.2H.A9GF.01A.11R.A37I.31"``); the result keeps
        the names in ``samples`` as index.
    """
    var = _variables()
    if names is None:
        names = list(var)
    names = [names] if isinstance(names, str) else list(names)
    unknown = [n for n in names if n not in var]
    if unknown:
        raise ValueError(f"Unknown covariate(s) {unknown}; available: {list(var)}")
    frames = {}
    for n in names:
        tname, d, loc, meta = var[n]
        key = (str(d), tname)
        if key not in frames:
            frames[key] = _read_table(d, tname, meta)
    out = pd.concat([frames[(str(var[n][1]), var[n][0])][n] for n in names], axis=1)
    if samples is not None:
        samples = list(samples)
        mapped = [sample_id(s) if sample_id else s for s in samples]
        out = out.reindex(mapped)
        out.index = samples
    return out


def add_covariates(name: str, table: pd.DataFrame, description: "str | Mapping[str, str]" = "", source: str = "",
                   overwrite: bool = False) -> Path:
    """Add a table of background variables to **your** registry (``$PYREA_DATA/covariates``, default ``~/.pyrea/covariates``).

    Parameters
    ----------
    name : str
        Name of the table.
    table : pd.DataFrame
        Samples (index: unique names, e.g. ``TCGA-XX-XXXX-01``) x numeric variables. The column names are the variable names for
        :func:`load_covariates` and must be unique across the registry (a prefix such as ``"estimate_"`` helps).
    description : str or dict
        One text for all variables or a dict variable -> text.
    source : str
        Where the values come from (reference, version, date); shown by :func:`available_covariates`.
    overwrite : bool
        Replace a table of your own with this name.

    Returns
    -------
    Path of the file.
    """
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("table must be a non-empty DataFrame (samples x variables)")
    if not name or not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError("name may contain letters, digits, '-', '_' and '.' only")
    if not table.index.is_unique:
        raise ValueError("The sample names (index) of table must be unique")
    if table.columns.duplicated().any():
        raise ValueError("The variable names (columns) of table must be unique")
    try:
        table = table.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All variables must be numeric (encode categories as 0/1 columns)") from exc
    tables = _tables()
    if name in tables and (tables[name][1] == "bundled" or not overwrite):
        raise ValueError(f"A table named {name!r} exists" + (" (bundled)" if tables[name][1] == "bundled" else "; use overwrite=True"))
    taken = {v for v, (t, *_rest) in _variables().items() if t != name}
    clash = sorted(set(map(str, table.columns)) & taken)
    if clash:
        raise ValueError(f"Variable name(s) already in the registry: {clash}; rename the column(s)")
    d = _user_dir()
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{name}.csv.gz"
    out = table.copy()
    out.index = out.index.astype(str)
    out.index.name = "sample"
    out.to_csv(d / fname, compression="gzip")
    idx = _read_index(d)
    desc = {str(c): (description.get(c, "") if isinstance(description, Mapping) else str(description)) for c in table.columns}
    idx["tables"][name] = {"file": fname, "source": source, "columns": desc}
    with open(d / "index.json", "w") as fh:
        json.dump(idx, fh, indent=1)
    return d / fname


def remove_covariates(name: str) -> None:
    """Remove a table of your own from the registry (bundled tables cannot be removed)."""
    d = _user_dir()
    idx = _read_index(d)
    if name not in idx["tables"]:
        raise ValueError(f"No table of your own named {name!r}" + (" (bundled tables cannot be removed)" if name in _tables() else ""))
    meta = idx["tables"].pop(name)
    f = d / meta["file"]
    if f.exists():
        f.unlink()
    with open(d / "index.json", "w") as fh:
        json.dump(idx, fh, indent=1)


# ---------------------------------------------------------------------- resolving the ways to give covariates
def _flatten(spec) -> list:
    if spec is None:
        return []
    if isinstance(spec, (pd.DataFrame, pd.Series, str, Mapping)):
        return [spec]
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(_flatten(s))
        return out
    raise ValueError("covariates must be a registry name, a gene list dict, a DataFrame / Series, or a list of these")


def resolve_covariates(spec, expression: pd.DataFrame, method: str = "mad", min_genes: int = 10,
                       species: str = "human") -> Optional[pd.DataFrame]:
    """One covariate table (samples x covariates, index = ``expression.columns``) from the ways to give covariates.

    ``spec`` is ``None``, the name of a registry variable (:func:`available_covariates`), a **dict of gene lists** ``{"EMT": genes, ...}`` that is
    scored per sample with :func:`score_genesets` on ``expression`` (needs R), a DataFrame / Series indexed by sample name, or a list of
    these. A single gene list needs a name (``{"EMT": genes}``): a list of strings is read as registry names. The columns of all parts must
    have different names. Registry variables are matched by the sample names of ``expression``; if none matches, an error says so (for TCGA
    map the names with ``sample_id=pr.tcga_sample_id`` first).
    """
    parts = _flatten(spec)
    if not parts:
        return None
    frames = []
    for p in parts:
        if isinstance(p, str):
            f = load_covariates([p], samples=expression.columns)
            if not f[p].notna().any():
                raise ValueError(f"No sample of the expression has a value for {p!r}; the sample names look like "
                                 f"{list(expression.columns[:2])}, the registry's like TCGA-XX-XXXX-01 (map them with sample_id=pr.tcga_sample_id)")
        elif isinstance(p, Mapping):
            f = score_genesets(expression, p, method=method, min_genes=min_genes, species=species)
        else:
            f = (p.to_frame() if isinstance(p, pd.Series) else p)
            if f.index.duplicated().any():
                raise ValueError("A covariate table has duplicated sample names")
            f = f.reindex(expression.columns)
        frames.append(f)
    out = pd.concat(frames, axis=1)
    dup = out.columns[out.columns.duplicated()].unique()
    if len(dup):
        raise ValueError(f"Covariate names repeat: {list(dup)}")
    out.index = expression.columns
    return out.astype(float)

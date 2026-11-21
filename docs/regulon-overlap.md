[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Overlap of regulons by tail (`regulon_overlap`)

Do two (or more) regulators share targets, and with the same sign of the mode of regulation? A regulon lists targets with a mode of
regulation `mor` in [-1, 1] (`tfmode` in `viper`): `mor > 0` activated, `mor < 0` repressed targets. `regulon_overlap` splits the regulon of
every regulator into these two tails and overlaps them per pair (no R needed):

```python
reg = pr.read_regulon_rds("FullReg.rds")                                    # source, target, mor, likelihood
ov = pr.regulon_overlap(reg, ["ZCCHC24", "ZEB1", "ZEB2"], tail="positive", n_perm=2000)
ov.table      # per pair and tail combination: n1, n2, overlap, expected, fold, jaccard, overlap_coef, odds_ratio, p_hyper, q_hyper, p_perm, q_perm
ov.pairs      # per pair, any sign: shared targets, concordant / discordant, agreement, Spearman correlation of mor
ov.sizes      # targets per regulator: all / positive / negative
ov.shared("ZCCHC24", "ZEB1", signs=("+", "+"))     # the shared targets with mor and likelihood of both
ov.matrix("jaccard", signs=("+", "+"))             # regulator x regulator matrix
```

- **`tail`** chooses which tails are overlapped: `"positive"` (default) the activated targets of both regulators (`+/+`), `"negative"` the
  repressed ones (`-/-`), `"both"` all four combinations, including the discordant ones (`+/-`: A activates what B represses).
  The sign threshold is `mor_min` (default 0: the sign alone; targets with `mor == 0` are in neither tail); `min_likelihood` drops weak
  interactions first. Any number of regulators (every pair is analysed, or `focus="TF"` against the others, or exactly `pairs=[("A", "B"), ...]`); `universe=` restricts the genes that can be targets (e.g.
  the genes of the expression matrix).
- **Statistics.** `expected` and `p_hyper` come from the hypergeometric distribution (random sets drawn from the universe; one-sided
  enrichment). ARACNe regulons share hub targets, which this null ignores and then overstates significance. With `n_perm` a
  **degree-weighted permutation null** is added: the second regulator's tail is redrawn with the same size, targets drawn with probability
  proportional to one plus the number of the *other* regulators that have them in the same tail (the two compared regulators are left out
  of the count); `perm_mean` is the mean overlap under this null, `p_perm` the one-sided p (smallest possible `1 / (n_perm + 1)`).
  `q_*` are Benjamini-Hochberg over all rows of `table`. Under no overlap the permutation p is not anti-conservative (simulated).
- **Example, STAD TCGA network.** ZCCHC24 (545 positive, 57 negative targets) and ZEB1 (351 / 26) share 87 positive targets (expected 11.4,
  7.6-fold; degree-weighted null 12.0) and 2 negative ones (expected 0.09); no discordant pair, all 89 shared targets have the same sign.

Figures follow the design of the others (clean panels, an `Axes` that can be passed, statistics added afterwards):

| panel | draw | add statistics |
| --- | --- | --- |
| regulator x regulator heatmap of one metric and tail combination | `ov.plot_heatmap(metric, signs)` | `ov.add_heatmap_values(ax=ax, fmt=...)` |
| shared targets of two regulators (likelihood, or `value="mor"`; concordant blue, discordant red) | `ov.plot_scatter(tf1, tf2, label_top=n)` | `ov.add_scatter_stats(tf1, tf2, ax=ax)`: overlap, expected, fold, p per tail combination and the sign agreement |
| intersections of the tails of all regulators (UpSet style; `sets=` selects) | `ov.plot_upset(max_bars=15)` | `ov.add_upset_counts(ax=ax)` |

`mor` is close to +-1 for most ARACNe targets, so the scatter shows the likelihood (interaction confidence) by default.

## Collecting the shared targets for export

The shared targets are kept in the result and can be gathered over analyses:

```python
ov.shared_targets          # ALL shared targets of ALL pairs, long: tf1, tf2, gene, mor1, mor2, likelihood1, likelihood2, sign1, sign2, concordant, in_tail
ov.gene_sets()             # {"ZCCHC24&ZEB1 (+/+)": [genes], ...} per pair and tail combination (signs=("+", "+") selects one)
ov.core("+", ["ZCCHC24", "ZEB1", "ZEB2"])        # targets in the positive tail of ALL given regulators

col = pr.GeneListCollection()                    # one place for lists from several analyses
ov_stad.collect(col, label="STAD TCGA")          # the lists of every pair (+ the core of three or more regulators), prefixed with the label
ov_acrg.collect(col, label="STAD ACRG")          # a second network into the same collection
col.add("manual EMT list", ["VIM", "CDH2"], source="curated")
col.summary()                                    # one row per list: size and provenance (tfs, tail, signs, overlap, expected, fold, p_hyper, p_perm, ...)
col.intersection(["STAD TCGA: ZCCHC24&ZEB1 (+/+)", "STAD ACRG: ZCCHC24&ZEB1 (+/+)"])      # replicated in both networks
col.export("shared_targets.xlsx")                # sheets: lists, genes (long); .gmt: gene sets for GSEA; any other path: a folder of CSV files

ov.export("overlap.xlsx", collection=col, enrichment=enr, split=sp)       # additionally: settings, table, pairs, sizes, shared_targets, enrichment, activity_correlation
```

## Over-representation of gene sets in the shared targets

`ov.enrich(genesets)` tests every shared-target list (and the core of three or more regulators) for over-representation of the terms of
gene set collections with `pr.gene_set_enrichment` (one-sided hypergeometric test, Benjamini-Hochberg within a target list; usable on its own for any
gene list). **The collections bundled with pyrea can be chosen by name**, no loading needed:

```python
pr.available_genesets()                               # collection, description, species, n_sets, size_min / median / max
enr = ov.enrich("h", signs=("+", "+"), min_size=10, min_overlap=3)                 # Hallmark
enr = ov.enrich(["h", "reactome", "pid", "wikipathways"])                          # several collections, tested together
enr[enr["q"] < 0.05][["target_set", "term", "collection", "overlap", "n_term", "fold", "p", "q", "genes"]]

pr.gene_set_enrichment(my_genes, "reactome", universe=expressed_genes)             # any gene list; species="mouse" for mouse symbols
ov.enrich({"my set": ["VIM", "CDH2", "FN1"]})                                      # or a dict of your own sets
```

- Collections: `"h"` (MSigDB Hallmark, also `"hallmark"`), `"reactome"`, `"pid"`, `"wikipathways"` (MSigDB v7.4, human and mouse symbols).
  With bundled collections the result has a `collection` column; with several, `q` is corrected over all their terms together.
- The default universe is the set of genes that could be targets of the network (`ov.universe_genes`); pass `universe=` (e.g. the expressed
  genes) to change it. A universe made of annotated genes only inflates enrichment.

## Does the overlap drive the correlation of the activities? (`split_activity`, needs R)

The activity of a regulator is inferred from its targets, so regulators that share targets correlate in activity partly by construction.
`split_activity` infers the activity of each regulator from three parts of its regulon: the **full** regulon, the **shared** targets (in
the chosen tail combination) and the **unique** targets (those the other regulator does not have at all), and correlates the parts of the two
regulators across samples:

```python
sp = ov.split_activity(expression, "ZCCHC24", "ZEB1", method="mad", n_random=200)
sp.correlation        # part1, part2, n1, n2, target_overlap, circular, rho, p (Spearman)
sp.matrix()           # parts of tf1 (rows) x parts of tf2 (columns)
sp.null_summary()     # shared part vs. the other's unique part against random subsets of the same size
sp.parts              # targets per part and whether VIPER could infer an activity (minsize)
sp.expression_corr    # correlation of the two regulators' own expression
sp.plot_correlations(); sp.add_correlation_values(ax=ax)      # matrix, circular cells hatched
sp.plot_scatter("unique", "unique"); sp.add_scatter_stats("unique", "unique", ax=ax)
```

- Cells whose parts share genes (`target_overlap > 0`, `circular=True`) are correlated by construction. The informative cells are those
  without shared genes: **unique x unique** (do the regulators vary together beyond the shared targets?) and **shared x unique** (do the
  shared targets behave like the other regulator's own targets, more than random targets of the first would? `null_summary()`).
- The signature is ranked over **all genes of the expression matrix** (`eset_filter=False`), as in an analysis with the full network. With
  the default filter of `viper_activity` the ranks would only cover the targets of the few regulons given here, and the parts would
  depend on each other (spurious negative correlations). `full x full` equals the correlation of the ordinary VIPER activities (tested).
- Caveat: when many genes shift together, VIPER's ranks of all other genes shift with them; use the random-subset null to see how far that carries.
- **Example (STAD, ZCCHC24 and ZEB1, method `mad`):** activities correlate with rho 0.93 (the regulators' expression with 0.86). Without any shared
  gene, `unique x unique` is still 0.87, and random ZCCHC24 targets correlate with ZEB1's unique targets at 0.88 +- 0.01 against 0.94 for the
  shared ones: the overlap of 87 targets adds little; both regulons follow one common axis.

## Many networks at once (`overlap_across_networks`)

For a list of networks (e.g. one ARACNe network per tumour type) the same analysis runs per network, **one network in memory at a time**,
and everything is collected in a `MultiOverlap`. Give the paths of the `.rds` files (or a dict `name -> path`); the pairs to test are
chosen as in `regulon_overlap`:

```python
paths = {"BRCA": "../Shared/networks/BRCA/ARACNe/FullReg.rds", "GBM": "../Shared/networks/GBM/FullReg.rds"}   # name -> path (or a plain list of paths)
mo = pr.overlap_across_networks(paths, ["ZCCHC24", "ZEB1", "ZEB2", "SNAI2"], n_perm=1000)                # all pairs of tfs
mo = pr.overlap_across_networks(paths, ["ZCCHC24", "ZEB1", "ZEB2", "SNAI2"], focus="ZCCHC24")            # one regulator against the others
mo = pr.overlap_across_networks(paths, pairs=[("ZCCHC24", "ZEB1"), ("SNAI2", "TWIST1")])                 # exactly these pairs

mo.summary()                  # per pair: in how many networks tested / a hit, median fold, jaccard, overlap, the networks with a hit
mo.matrix("fold")             # networks x pairs (any column of mo.table; signs=("+", "+") chooses the tail combination)
mo.table, mo.pairs, mo.sizes, mo.shared_targets       # all networks, with the column `network`
mo.coverage, mo.skipped       # per network: path, pairs tested, missing regulators; networks that cover no pair
mo.results["GBM"]             # the RegulonOverlap of one network (figures, enrich, split_activity)
```

- **Pairs.** `tfs` alone: all pairs; `focus` (one of `tfs`): that regulator against each of the others (n - 1 instead of n (n - 1) / 2
  tests, and less permutation time); `pairs`: exactly these. A network without some of the regulators is analysed for the pairs it covers
  (`coverage`), one that covers none is skipped with a warning.
- **Names.** Prefer a dict `name -> path`: the key is the `network` column and the key of `mo.results`, and the result can be merged by it
  (below). With a plain list the name is the file name without extension; if the files are all called `FullReg.rds`, the folders that tell them
  apart (`"BRCA/ARACNe"`, `"GBM"`); `names=[...]` or a dict overrides. Items may also be regulon DataFrames (then names are needed) and
  `reader=` reads other formats than `.rds`.
- **Memory and speed.** Each network is read, analysed and released; only the small results (the requested regulators) stay. `n_jobs > 1`
  analyses that many networks in parallel in separate processes (each reads its own file and holds one network; R is started in each).
  The permutations and reading the `.rds` dominate the run time. `cache_dir=` stores the result per network (key: the file with size and
  modification time, and all settings including the pairs, so other settings or pairs recompute the network).
- **Multiple testing.** `q_hyper` / `q_perm` are per network (over its pairs); `q_hyper_all` / `q_perm_all` are Benjamini-Hochberg over **all** rows
  of all networks and are what `summary()` and `hits()` use. A *hit* is `q_*_all < q` (default 0.05), `fold >= fold_min` (2) and
  `overlap >= min_overlap` (3); `p="perm"` / `"hyper"` chooses the null (default: permutation if run). The networks are not independent
  (related tumour types, overlapping samples): "hit in 12 of 20 networks" describes, it is not a test of consistency.
- **Merging.** `mo_a.merge(mo_b)` (or `pr.MultiOverlap.concat([...])`) combines analyses of *different* networks by name, e.g. cohorts added
  later or runs with other pairs. It needs equal `tail`, `mor_min`, `min_likelihood`, `n_perm` and `seed` and no repeated network name (errors
  otherwise); `q_*_all` are recomputed over all rows of the combined table, because the correction depends on the number of tests. The result
  equals a joint run (tested).
- **Recurrent targets and export.**

```python
rec = mo.recurrent_targets(min_networks=5)               # pair, gene, n_networks, n_tested, frac, mean likelihoods, networks
col = mo.collect(min_networks=5)                         # GeneListCollection: one list per pair; per_network=True adds each network's lists
enr = mo.enrich("h", min_networks=5)                     # over-representation (bundled collection by name or your own dict)
mo.export("results/ZCCHC24_pancancer.xlsx", min_networks=5, enrichment=enr)     # or .gmt / a folder of csv files

ax = mo.plot_matrix("fold"); mo.add_hits(ax=ax)          # networks x pairs heatmap; hits marked afterwards
```

  The default universe for `enrich` is, per pair, the intersection of the target universes of the networks in which it was tested.

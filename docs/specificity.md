[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Is a set of regulators more closely linked than other regulators? (`regulon_specificity`)

`regulon_overlap` asks whether two regulators share more targets than random *genes* would. That does not show that they are special:
regulators of related functions share hub targets, and large regulons overlap more by chance. `regulon_specificity` uses the other
**regulators of the same network** as the reference, for pairs and for any larger set (no R needed):

```python
sp = pr.regulon_specificity(reg, [("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "ZEB2")], tail="positive", n_random=5000)
sp.table       # per set and tail combination: sizes, overlap, expected, fold, null_median_fold, null_p95_fold, percentile, p_random, ext_rank_max, mutual_best
sp.extension   # per set and member: rank of the member among ALL regulators as a partner of the others (+ the top candidates)
pr.top_partners(reg, "ZCCHC24", top=20)          # the regulators that overlap most with ZCCHC24 (or with the common targets of several)
```

- **Size-matched random sets** (`p_random`): every member is replaced by a random regulator whose tail has about the same size
  (`tol`, `min_pool`) and the fold = overlap / expected of the resulting sets is compared (`n_random` sets; smallest p `1 / (n_random + 1)`).
  `percentile` is the share of random sets with a smaller fold. The null keeps the hub structure of the network, which the gene-level null of
  `regulon_overlap` ignores (tested: an overlap that is "significant" against random genes is not unusual among regulators).
- **Partner ranks** (`extension`): the other members form the base (their common targets), and all regulators with a tail of at least `min_size`
  targets are ranked by the hypergeometric p of their overlap with it. For a pair this is the rank of B among the partners of A and of A among the
  partners of B (`mutual_best`: both 1); for a triplet, how good C is as the third member given A and B. `ext_rank_max` /
  `ext_percentile_max` (rank / candidates) is the weakest link of the set; `top` lists the best candidates with their overlap.
- **Tails.** `tail="positive"` (default), `"negative"` or `"both"`: every combination of the members' tails (`+/+`, `-/-`, `+/-`, `-/+`; 8 for
  a triplet), where anti-correlated regulators show up. Only regulators with a tail of at least `min_size` (25) are in the reference; with no
  such regulator for a sign `p_random` is NaN.
- **Reading the result.** Real networks are modular: most random regulator pairs share nothing, but regulators of one module overlap strongly. A
  set can then be clearly above random regulators (small `p_random`) and still not be the tightest pair of its module (partner rank 10 of 1300);
  the ranks show which of the two holds. `p_random` says "more than regulators of that size usually do", not "related by function".

Across many networks the same analysis runs with `specificity_across_networks` (networks as paths or a dict `name -> path`, names, `n_jobs`,
`cache_dir`, `reader`, merging by name as for `overlap_across_networks`):

```python
ms = pr.specificity_across_networks(net_paths, [("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")], n_random=5000, n_jobs=4)
ms.summary()                      # per set: networks evaluated / hits, median fold vs matched null, median percentile, n_mutual_best, n_top
ms.matrix("percentile")           # networks x sets (also "fold", "p_random", "ext_percentile_max"; signs="+/-" for a tail combination)
ms.extension                      # partner ranks and top candidates per network
ax = ms.plot_matrix("percentile"); ms.add_hits(ax=ax)
ms.merge(other); ms.export("results/specificity.xlsx")
```

A *hit* needs `q_random_all` < 0.05 (Benjamini-Hochberg of `p_random` over all rows of all networks), `fold >= 2` and `overlap >= 3`. Because of the
floor `1 / (n_random + 1)` many rows at the floor are needed before the BH correction lets any pass; use a large `n_random`. `n_top` (networks in
which every member is within the best `top_frac`, 5 %, of the partners of the others) has no such floor. The networks are not independent.

# Are the activities of a set of regulators more closely linked than those of other regulators? (`activity_specificity`)

The activity counterpart of `regulon_specificity`: instead of the overlap of the targets, the Spearman correlation of the VIPER activities
across the samples of a cohort, again with the **other regulators of the network** as reference (pure numpy; the activity matrix comes from
`viper_activity(..., eset_filter=False)` over the **whole network**, so that the ranks of the signature do not depend on the few
regulons given):

```python
act = pr.viper_activity(expression, reg, method="mad", minsize=25, eset_filter=False)
sp = pr.activity_specificity(act, [("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")], covariates=purity)
sp.pairs        # every pair x activity variant: rho, p and (with covariates) partial_rho, partial_p
sp.table        # per set: rho_mean, abs_rho_min, partial versions, ext_rank_max / ext_percentile_max / mutual_best (raw and padj_*)
sp.extension    # per set, member and adjusted: rank among ALL regulators as a partner of the other members, percentile, top candidates
pr.activity_partners(act, "ZCCHC24", covariates=purity, top=20)      # whose activity is most closely linked to ZCCHC24's?
```

- **Partner ranks.** The other members form the base; every regulator of the network is regressed on the base activities and ranked by R²
  (for a pair: by the squared correlation; `direction="positive"` / `"negative"` ranks by the signed correlation, which is what an anti-correlated
  pair needs when only the negative direction is of interest). `ext_percentile_max` (rank / candidates of the weakest member; small = good) and
  `mutual_best` are as in `regulon_specificity`.
- **Covariates** (e.g. tumour purity): the ranks of the activities are regressed on the ranks of the covariates; the partial Spearman correlation
  is the correlation of the residuals (p-value with `n - 2 - q` degrees of freedom), and the adjusted ranks are those of the residuals. Samples without
  a covariate are dropped from **all** statistics (`settings["n_dropped"]`). The adjustment is linear in the ranks: it removes a monotone purity effect,
  not every confounder.
- **Activity variants.** `variants={"pleiotropy": ...}` adds pair correlations from further activity matrices, e.g. VIPER activity from a regulon
  **restricted to the candidates** with the pleiotropy ("shadow") correction (`viper_activity(..., pleiotropy=True, pleiotropy_args=dict(regulators=...,
  shadow=..., targets=..., penalty=..., method="adaptive"|"absolute"))`), which down-weights targets shared with a more significant regulator. How much
  of the correlation lives in the shared targets is the difference to the full activity. Note that VIPER's R code fails when, in a sample, only a
  single pair of significant regulons shares enough targets (an error message says so); with a handful of overlapping candidates it runs.

Across cohorts (one expression matrix and network each; needs R):

```python
cohorts = {"BRCA": (net_paths["BRCA"], "TCGA-BRCA-logCPM.tsv"), "STAD": (net_paths["STAD"], "TCGA-STAD-logCPM.tsv")}   # name -> (network, expression)
ma = pr.activity_specificity_across_cohorts(cohorts, [("ZCCHC24", "ZEB1")], covariates="absolute_purity", sample_id=pr.tcga_sample_id,
                                            pleiotropy=True, n_jobs=4, cache_dir="results/activity_cache")      # covariates=None: no adjustment (default)
ma.coverage; ma.summary()          # per set: cohorts, median rho, median partner percentile, n_top (every member within the best 5 %), n_mutual_best
ma.meta(adjusted=True)             # random-effects meta-analysis of every pair (Fisher z, Spearman variance 1.06): pooled rho, CI, I2
ma.pair_matrix("partial_rho"); ma.pleiotropy_effect()
ax = ma.plot_matrix(); ma.add_top(ax=ax); ma.plot_forest("ZCCHC24", "ZEB1", adjusted=True)
ma.merge(other); ma.export("results/activity.xlsx")
```

- **Covariates are optional** (`covariates=None`, no adjustment, is the default) and can be given as a registry name, a dict of gene lists that is
  scored per cohort, a table, or a list of these (see [covariates](covariates.md)).
- Networks and expression are read per cohort (paths or DataFrames; `expression_reader=` / `regulon_reader=` for other formats), the activity
  of the whole network is inferred in each, and covariates are matched by sample name; `sample_id` maps the expression's column names to the
  covariate index (`pr.tcga_sample_id`: `"TCGA.2H.A9GF.01A.11R.A37I.31"` -> `"TCGA-2H-A9GF-01"`; with `n_jobs > 1` the function must be picklable, i.e.
  defined in a module). Names that repeat after the mapping keep the first sample (`coverage["n_duplicated_samples"]`).
- If VIPER's pleiotropy correction fails in a cohort the other results stay and `coverage["pleiotropy"]` says `failed: ...`.
- The pooled correlation averages different tumour types: look at `I2` / `p_heterogeneity` (and the forest plot) before quoting one number.
  `n_jobs` cohorts run in parallel in separate processes, each with its own R. Caching and merging by name work as for `overlap_across_networks`.
- **Example (23 TCGA cohorts, TF networks, ABSOLUTE purity; ZCCHC24 | ZEB1).** Pooled Spearman rho 0.75 (95 % CI 0.64-0.83, `I2` 0.99), 0.73 after
  adjusting for the purity; the correlation ranges from -0.5 (LAML) to 0.96 (ESCA). The weakest member is within the best 5 % of the partners of the
  other in 13 of 23 cohorts, never the single best. The pleiotropy correction lowers rho by 0.03-0.12 (e.g. STAD 0.94 -> 0.85).

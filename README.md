# pyrea

Gene set enrichment tools: single- and two-tailed GSEA, VIPER/aREA regulon
activity, and the plotting helpers that go with them. Extracted from a
working analysis script.

Enrichment scoring (`aREA`, and therefore `Viper`, `Gsea1T`, `Gsea2T` and
friends) is computed by the Bioconductor package
[`viper`](https://bioconductor.org/packages/viper/) via `rpy2` -- not a
Python reimplementation -- so scores are identical to running `viper::aREA`
in R.

## Install

```bash
pip install -e .            # core: gene set / regulon utilities, no R needed
pip install -e '.[r]'       # aREA / Viper / Gsea1T / Gsea2T (needs R + Bioconductor viper)
pip install -e '.[r,test]'
```

```r
BiocManager::install("viper")
```

## Layout

| Module | Contents |
| --- | --- |
| `pyrea.rbackend.area` | `aREA` -- analytic rank-based enrichment (Alvarez et al., Nat Genetics 2016), via `viper::aREA` (needs the `r` extra) |
| `pyrea.Gsea1T` | `Gsea1T`, `Gsea1TMultSets`, `Gsea1TMultSigs` -- single-tailed GSEA |
| `pyrea.Gsea2T` | `Gsea2T`, `GseaReg`, `GseaRegMultSigs`, `GseaMultReg` -- two-tailed / regulon GSEA |
| `pyrea.Viper` | `Viper` -- VIPER regulon activity inference |
| `pyrea.utils` | `gene_sets_to_regulon`, `sig_to_reg`, `load_genesets`, `load_species_converter` |
| `pyrea.plotting` | internal plotting helpers used by the classes above |
| `pyrea.mps` | Mutant phenotype score (Alvarez et al. 2016): `relative_likelihood`, `mutant_phenotype_score`, `classify_phenotype`, `mutation_association`, `locus_specific_mps`. Only the association tests need R |
| `pyrea.rbackend.viper` | `viper_activity` -- `viper::viper` (mad/rank/... signatures) for protein activity, as DataFrame (needs R) |
| `pyrea.rbackend.regulon` | `read_regulon_rds` -- ARACNe/viper regulon from `.rds` to long format (needs R) |
| `pyrea.rpt` | `viper_rpt` -- residual post-translational activity (`rank`, `lineal`, `spline`), pure Python, matches `viper::viperRPT` |
| `pyrea.mps_plot` | `plot_mps_rank`, `plot_mps_traits`, `plot_rl_diagnostic` |
| `pyrea.concordance` | `activity_vs_expression` -- is an effect seen in the activity but not (or oppositely) in the expression? |

```python
reg = pr.read_regulon_rds("FullReg.rds")
vpres = pr.viper_activity(emat, reg)                         # method='mad'
rpt = pr.viper_rpt(vpres, emat)                              # method='rank'
mps = pr.mutant_phenotype_score(mutations, vpres, rpt)       # genes x samples, in [-1, 1]
pr.classify_phenotype(mps.loc["ERBB2"])                       # mutant / wt / intermediate (LR > 3)
assoc, var_mps = pr.locus_specific_mps(variants, vpres, rpt, expression=emat)  # rows "GENE:variant"
```

The public API is re-exported at the top level:

```python
import pyrea as pr

regulon = pr.gene_sets_to_regulon(pr.load_genesets("h", "human"))
res = pr.Gsea1T(signature, regulon)
```

## Bundled gene sets

`pyrea/data/` ships MSigDb v7.4 collections (Hallmark, PID, Reactome,
WikiPathways; human + mouse symbols) plus a human/mouse homology map, loaded
via `load_genesets(collection, species)` / `load_species_converter(keys)`.

## Mutant phenotype score and locus-specific variants

Implements the mutant phenotype score (MPS) of
[Alvarez et al., Nat Genet 2016](https://doi.org/10.1038/ng.3593): how much does a sample's
VIPER activity of a protein resemble that of the samples with a mutation in its gene, compared
with wild type?

```python
import pyrea as pr

vpres = pr.viper_activity(emat, regulon)          # global activity (G)
rpt = pr.viper_rpt(vpres, emat)                   # residual post-translational activity

# genes x samples, 1 = mutated, 0 = wild type, NaN = not profiled (excluded from both groups)
mps = pr.mutant_phenotype_score(mutations, vpres, rpt)       # genes x samples, in [-1, 1]
pr.classify_phenotype(mps.loc["ERBB2"])                       # mutant / wt / intermediate (LR > 3, RL > 0.5)
pr.plot_mps_rank(mps.loc["ERBB2"], mutations.loc["ERBB2"])   # samples ranked by MPS, carriers marked
```

#### RPT methods

`viper_rpt(vpres, emat, method=...)` removes the part of a protein's activity that follows the expression of its
gene. The three methods of `viper::viperRPT` are available (no R needed); sample names must be unique (collapse
duplicates first, e.g. `emat.T.groupby(level=0).mean().T`).

| `method` | model | note |
|---|---|---|
| `"rank"` (default) | linear regression of the within-gene ranks | as in the paper; robust to outliers |
| `"lineal"` | linear regression of the raw values | `viperRPT`'s linear option |
| `"spline"` | cubic smoothing spline (`spar=1.2`, a knot per distinct value) | removes non-linear dependence; R's default; ~15 s for 5000 proteins |

On STAD/EAC (ERBB2 amplification, MYC, ...) and GBM (EGFR, PTEN, ...) the RPT of the three methods correlates with
Spearman ~0.94 (median over proteins), the MPS with ~0.75-0.9. Strong signals are stable (EGFR in GBM: 68 / 67 / 67 %
of the carriers reach the mutant phenotype with rank / lineal / spline), weaker ones shift: ERBB2 amplification in
STAD gives 55 / 44 / 40 %. The reason is the combination step "larger |RL| of G and RPT": where RPT wins the MPS can
jump between the two curves, and how often that happens depends on the RPT method (`lineal` most often). Use the
default for results comparable with the paper and check others with `plot_mps_traits` (below).

#### Smoother or more transparent rank plots

The rank-sorted MPS is a sigmoid for a gene with a clear, one-sided effect (EGFR in GBM, Alvarez et al., Fig. 5) but can
show plateaus and steps otherwise. Two things cause it:

- *saturation*: with Scott's bandwidth the RL reaches +-1 in the tails of the densities (about 5 % of the samples
  for ERBB2 / EGFR). `bw_adjust` multiplies the kernel bandwidth (`bw_adjust=1.5-2` removes the saturation);
  larger values flatten the curve and lower the share of samples that reach RL > 0.5. It is accepted by
  `relative_likelihood`, `mutant_phenotype_score`, `locus_specific_mps`, `mps_targets`, `compare_phenotypes` and
  `plot_rl_diagnostic`; the default 1.0 reproduces the previous results.
- *steps between G and RPT*: `plot_mps_traits` draws the RL of G, of RPT and the combined MPS side by side, so
  the trait that causes a jump is visible.

```python
mps = pr.mutant_phenotype_score(mutations, vpres, rpt, bw_adjust=1.5)
fig = pr.plot_mps_traits("EGFR", mutations, vpres, rpt, bw_adjust=1.5)   # panels: G, RPT, MPS
```

Rows that are not genes (e.g. a gene fusion) are scored on the activity of one or more chosen
proteins:

```python
summary, mps = pr.mps_targets(fusion_matrix, ["RHOA", "PTK2", "YAP1"], vpres, rpt, expression=emat)
```

### Individual variants (locus-specific)

Different variants of one gene can act differently (e.g. *KRAS* G12D vs G13D, nonsense vs
missense *TP53*). `locus_specific_mps` takes a matrix of variants (rows `"GENE:variant"`):

```python
import cbiokit as cbk

ex = cbk.read_alteration_export("alterations_across_samples.tsv")
variants = cbk.alteration_matrix(ex, types=("MUT",), level="event",
                                 by="PATIENT_ID")[emat.columns]          # all variants, VUS included
assoc, mps = pr.locus_specific_mps(variants, vpres, rpt, expression=emat,
                                   min_samples=3, min_mps=10)

assoc.loc[assoc.index.str.startswith("KRAS:")].sort_values("p_min")
pr.plot_mps_rank(mps.loc["TP53:R175H"], variants.loc["TP53:R175H"])
```

Things to know:

- **Wild type is the gene's wild type.** In a variants matrix a sample with a *different*
  variant of the same gene is also 0, but it is not wild type. `locus_specific_mps` first
  derives the gene state (any variant / none / not profiled). The MPS densities use samples
  without any mutation in the gene as WT; the association excludes carriers of other variants
  (`exclude_other_variants=True`, count in `n_other_variants`; set `False` to test against all
  other samples).
- **Pass all variants of a gene.** Carriers of a variant you filtered out beforehand would look
  like wild type. Filter with `min_samples` / `min_mps`, not before.
- **Include variants of unknown significance** (the default of `cbiokit.alteration_matrix`; `ex.drivers()` restricts to drivers); the paper uses
  all non-silent mutations, not only annotated drivers.
- Few carriers mean noisy tests: `min_samples=2` (paper) yields many p-values, and `p_min` (the
  smaller of G and RPT) is not corrected for multiple testing. The MPS itself needs at least
  `min_mps` carriers (default 10).
- The pan-cancer integration over tumour types (paper Fig. 6b) is not implemented.

### Do two alterations act alike? (phenocopy)

`compare_phenotypes` compares alteration groups on the activity of one protein, e.g. whether a
gene fusion mimics a mutation of the target gene:

```python
groups = {"RHOA mut": mut.loc["RHOA"], "ARHGAP fusion": fus.loc["CLDN-ARHGAP"]}   # 1 / 0 / NaN per sample
cmp = pr.compare_phenotypes(groups, "RHOA", vpres, rpt)

cmp.summary      # (group, score): n, mean, median, auc (+ interval) and p against the reference (G, RPT, MPS scales)
cmp.pairwise     # group vs group
cmp.cross        # how do the samples of one group score on the MPS scale defined by the other?
cmp.scores       # per sample (group label, G, RPT, MPS[<group>]) for plotting
```

- The **reference** is every sample that is 0 in *all* groups, so carriers of the other alteration are
  not wild type. Samples in several groups ("overlap") and samples not profiled for a group are left
  out. Each group is compared on its own samples only.
- `cross` is the phenocopy test: scale A is defined by A vs reference; B was not used for that scale,
  so B scoring high on it (`frac_mutant_phenotype`, `auc`, `p`) means B looks like A. Check both
  directions. Where scale == group (`in_sample=True`) the value is optimistic.
- `auc` is the probability that a group sample has a higher value than a reference sample
  (0.5 = no difference); `p` is a two-sided Mann-Whitney test, not corrected for multiple testing.

#### How to read `cmp.cross` (the phenocopy test)

Question: **does alteration B put the protein into the same state as alteration A?** (A, B: e.g. an activating
mutation and an amplification, a hotspot and a rare variant, a known driver and a gene fusion.)

1. *Scale of A.* From the activity of the target in the A samples and in the reference (samples with none of the
   alterations) the relative likelihood `RL(x) = (p_A(x) - p_ref(x)) / (p_A(x) + p_ref(x))` is estimated for every
   activity value (G and RPT, the one with the larger |RL| counts). RL > 0 means: more like A than like the
   reference; RL > 0.5 is a likelihood ratio above 3 (the "mutant phenotype" of Alvarez et al.).
2. *Score B on it.* All samples of B receive their RL on the scale of A. B did not take part in estimating that scale,
   so a high score is evidence and not circular. Reported per `(scale, group)`: `n`, `mean_rl`,
   `frac_mutant_phenotype` (share with RL > 0.5), `auc` (probability that a B sample scores higher than a reference
   sample; 0.5 = no difference) and `p` (Mann-Whitney).
3. *Both directions.* The same with the scales swapped. Rows with `in_sample=True` (group == scale) show how well A
   separates from the reference by construction; they are optimistic and serve as the benchmark.

| result | reading |
| --- | --- |
| B high on A's scale **and** A high on B's scale | mutual phenocopy: same functional state |
| B high on A's scale, A not on B's | asymmetric: B resembles A, but A is weaker or more heterogeneous than B (e.g. amplification vs. mutation: the amplification looks "like the mutation, only stronger") |
| B not above the reference on A's scale | no phenocopy: B does something else, or nothing, to this protein |
| `in_sample` row low | A itself has no clear phenotype on this protein: nothing to copy |

`cmp.pairwise` answers the complementary question (do A and B differ from each other?). A result that looks
like a phenocopy in `cross` and shows no difference in `pairwise` is the strongest case.

#### Where the AUC comes from

The `auc` columns (and the Cliff's delta of `activity_vs_expression`, `delta = 2 * AUC - 1`) are the **probability that a
random sample of one group has a higher value than a random sample of the other group** (ties count half):

```
AUC = P(X_group > X_other) + 0.5 * P(X_group = X_other)  =  U / (n_group * n_other)
```

`U` is the Mann-Whitney statistic, so the AUC and the Mann-Whitney `p` next to it are two views of the same test. It is also
the area under the ROC curve that results when the value is used as a classifier "group vs. other" (0.5 = no separation,
1 = every group sample above every reference sample, 0 = all below). It uses the *ranks* of the two distributions only: no
density is estimated, there is no bandwidth, and it does not change under monotone transformations of the values.

Where it appears in `compare_phenotypes`:

| table | compares | on which values |
| --- | --- | --- |
| `summary` | group vs. reference | G, RPT and every group's MPS scale |
| `pairwise` | group A vs. group B | the same scores |
| `cross` | group B vs. reference | the MPS scale defined by group A (B was not used to build it) |

How it differs from the MPS (relative likelihood):

- The RL is a *local* value per sample, computed from the two kernel densities at that sample's value; it can be non-monotone
  (a gene whose mutated samples are bimodal). The AUC is *one number per comparison* and asks only whether the group is shifted
  up or down as a whole. A bimodal group with one mode on each side of the reference gives an AUC near 0.5 although the distributions
  differ; the RL (and `frac_mutant_phenotype`) would show it. Look at both.
- On G or RPT alone the AUC does not depend on `bw_adjust`. On the MPS scale it does, because the scale is built from the KDEs
  and takes the larger |RL| of G and RPT (not a monotone function of either).
- With few samples the AUC is uncertain, so every table has a bootstrap interval, `auc_low` / `auc_high` (95 %,
  `alpha=0.05`). Each group and the reference are resampled separately (`n_boot=1000`, `seed=0`; `n_boot=0` switches it off
  and gives NaN). For the MPS scales the scale itself is estimated again in every resample from the resampled group and
  reference, so the uncertainty of the scale is included and the interval is not too narrow. The `p` is the Mann-Whitney
  test of the original data, not corrected for multiple testing. Runtime: a few seconds for ~500 samples and two
  groups (the cost grows with the number of groups, one MPS scale each).
  For small groups (a rare fusion, n < 15) also show the individual points next to the AUC (`cmp.scores`).

```python
cmp = pr.compare_phenotypes(groups, "RHOA", vpres, rpt)            # intervals included
cmp.cross[["n", "frac_mutant_phenotype", "auc", "auc_low", "auc_high", "p"]]
```

**What it is good for.** (1) *Functional annotation of non-canonical alterations*: does a fusion, a rare variant or a
variant of unknown significance perturb the same protein as the established driver (the RHOA mutation vs.
CLDN18-ARHGAP fusion example)? (2) *Convergence of different alteration types* on one node: amplification and
mutation of ERBB2, truncating and missense CDH1 variants, deletion and mutation of a tumour suppressor. (3)
*Hypotheses for patient selection*: samples without the actionable alteration but with the same protein state could
be candidates for the same targeted drug (the idea behind the wild-type samples with a high MPS in Alvarez et al.,
Fig. 5a). (4) *Heterogeneity*: carriers that do not show the phenotype are candidates for passengers or subclonal
alterations.

**Limits.** The scores describe the *state* inferred from the expression of the regulon, not causality; only a
perturbation or an orthogonal measurement (protein, phospho-protein, drug response) establishes it. Co-amplified
neighbours can contribute to the activity of an amplified gene (compare G and the expression-adjusted RPT). Groups
of a few samples give noisy scales; with several targets and comparisons nothing is corrected for multiple testing.
The reference is shared by all scales, so differences between scales are not caused by different reference
samples. `compare_phenotypes` is an analysis built on the MPS of Alvarez et al., not a published or validated method.

### Does the effect differ between subtypes? (stratified comparison)

Pooling subtypes mixes the effect of an alteration with baseline differences between them (VIPER
activity with `method="mad"` is relative to the whole cohort). Compare mutants with wild types
*within* each stratum, then test whether the effect differs:

```python
res = pr.stratified_comparison(arid1a, subtype, "ARID1A", vpres, rpt, expression=emat)

res.effects        # (stratum, trait): n_mut, n_wt, auc, bootstrap 95 % interval, Mann-Whitney p
res.interaction    # p of the mutation x stratum term (rank-based linear model), per trait
res.pairwise       # AUC difference between two strata with bootstrap interval and p

diff = pr.differential_activity(arid1a, vpres, subtype)    # the same for every regulator
diff.sort_values("interaction:p").head()                   # <stratum>:auc/p/q, interaction:p/q
```

- `auc` is the probability that a mutant has a higher value than a wild type *of the same stratum*
  (0.5 = no effect). Read it with its interval: with few mutants the intervals are wide.
- Two separate p-values (significant in one stratum, not in the other) are no evidence for a
  difference; use `interaction` / `pairwise`.
- Strata need `min_group` mutants and wild types (default 3); others are left out. `group` is
  1 / 0 / NaN as everywhere (NaN = neither mutant nor wild type).
- Nothing is corrected for multiple testing except the `q` columns of `differential_activity`
  (Benjamini-Hochberg over the regulators). Under random labels the p-values are uniform.

### Does activity show something that expression does not? (`activity_vs_expression`)

A recurring question: does the VIPER activity of a gene show an effect of an alteration that its own expression
does not, or one pointing the other way? `activity_vs_expression` makes the same two-group comparison on expression,
G and (optionally) RPT and compares the effects directly:

```python
res = pr.activity_vs_expression(mutations, vpres, emat, rpt=rpt)     # rows like mutant_phenotype_score
res.sort_values("abs_diff")[["delta_expr", "delta_G", "delta_RPT", "abs_diff", "abs_low", "abs_high", "category"]]
```

- The effect on each scale is **Cliff's delta** (`2 * AUC - 1`, in [-1, 1], > 0 = higher in the altered samples). It is rank-based
  and therefore comparable between expression (log counts) and activity (NES-like).
- "Significant on one scale, not on the other" does not show a difference. The **difference** of the effects is
  estimated directly with a paired bootstrap over the samples (the scales are measured on the same samples and are
  correlated, since activity is computed from expression): `delta_diff = delta_G - delta_expr` and
  `abs_diff = |delta_G| - |delta_expr|` (is the activity effect larger or smaller than the expression effect?), each with a
  `1 - alpha` interval.
- `category`: `concordant`, `opposite`, `activity_only`, `expression_only`, `unclear` (only one scale shows an effect, the
  difference is not established), `none`. An effect means BH `q < alpha` (per scale, over all rows) and
  `|delta| >= min_effect` (default 0.2). The categories sort results; the effects and intervals are the result.
- With `rpt`, `beyond_expression` marks rows where RPT shows an effect in the same direction as G.

```python
# scatter of the effects with cbrviz (colour = category, dotted diagonal = same effect on both scales)
import matplotlib.pyplot as plt
from cbrviz import XYview

cats = [c for c in pr.CONCORDANCE_CATEGORIES if c in set(res["category"])]
xv = XYview(res[["delta_expr", "delta_G"]], s1=res["category"], s1_order=cats,
            s1_colors=[pr.CONCORDANCE_COLORS[c] for c in cats], highlight=list(res.index), s=30, alpha=0.9)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.scatter(**xv.scatter_call)
ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Effect on expression", ylabel="Effect on activity (G)")
xv.add_xy_line(ax=ax)
xv.label_dots(ax=ax, adjust=True, fontsize=7)
ax.legend(handles=xv.handles, frameon=False, fontsize=6)
```

pyrea itself does not depend on cbrviz; the table is all a plot needs. (`XYview.label_dots` needs column names that are valid
identifiers, so keep `delta_expr` / `delta_G` and set the axis labels afterwards.)

Reading the result (examples from TCGA STAD/EAC amplifications and GBM drivers):

- An amplified gene usually shows a **larger effect on expression than on activity** (`abs_diff < 0`, mostly
  `expression_only`): the copy number raises the transcript directly, the regulon follows less. ERBB2 amplification is
  concordant (delta_expr 0.95, delta_G 0.64). Activity larger than expression is rare (VEGFA: 0.89 vs 0.82, RPT +0.49).
- In GBM, NF1 alterations lower the expression strongly (-0.85) but not the activity (-0.16): `expression_only`.
- **RPT is not independent of this difference**: it is the residual of the activity after regressing on expression, so where the
  activity does not follow a large expression effect, RPT takes the opposite sign (NF1: RPT +0.63) by construction. Read
  `beyond_expression` together with the effect on G, not on its own.
- A signal only in the activity needs checks before it is believed: copy-number effects on neighbouring regulon genes (compare
  G and the amplification itself), subtype composition (`stratified_comparison`), and replication in another cohort.

## Clustering samples by protein activity

`pyrea.cluster` follows the *Unsupervised data analysis* of
[Alvarez et al., Nat Genet 2018](https://doi.org/10.1038/s41588-018-0138-4):

```python
res = pr.cluster_samples(vpres)                       # VIPER similarity, PAM, k from the cluster reliability
res.k, res.scores                                     # chosen k; global reliability for every k tried
res.labels.value_counts()                             # cluster 1..k, largest first
res.reliability                                       # per patient: nes and scaled reliability
pd.crosstab(res.labels, subtype)

pr.cluster_samples(pr.zscore_signature(expr), similarity="correlation")   # expression instead (Pearson)

emb = pr.tsne_embedding(vpres, perplexity=40, n_iter=5000)                # needs scikit-learn (pyrea[cluster])
pr.ari_permutation_test(res.labels, other_labels, n_perm=10000)           # adjusted Rand index and p
```

Building blocks: `zscore_signature`, `viper_similarity` (= `viper::viperSimilarity`),
`similarity_to_distance`, `pam` (= `cluster::pam`: BUILD + SWAP), `cluster_reliability`,
`adjusted_rand_index`. `viper_similarity`, the distance and `pam` are checked against R in the tests.

- **Cluster reliability.** For every sample the members of its cluster are a gene set and the
  negative distances to all other samples the signature; the aREA NES says how close the cluster
  mates are. The scores are scaled to [0, 1] and averaged (= area over the cumulative curve) per
  cluster and overall; k is the first local maximum of the global value.
- **Neither the manuscript nor its supplement says how the scores are scaled** (Supplementary Fig. 2d-g only
  shows the result). Both options are kept in `res.scores`; compare them before trusting k. The default (`scale="bounds"`) uses the
  smallest and largest NES possible for a cluster of that size, so that 0.5 is chance level and
  values are comparable between k. `scale="minmax"` scales within the partition; then the global
  value hardly tells good from random partitions (a synthetic test shows it).
- With continuous data the global reliability keeps rising with k and the first local maximum can
  be fragile: look at `res.scores` before using the clusters.


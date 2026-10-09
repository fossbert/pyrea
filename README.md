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
| `pyrea.rpt` | `viper_rpt` -- residual post-translational activity, pure Python, matches `viper::viperRPT` |
| `pyrea.mps_plot` | `plot_mps_rank`, `plot_rl_diagnostic` |

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
                                 drivers_only=False, by="PATIENT_ID")[emat.columns]
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
- **Include variants of unknown significance** (`drivers_only=False` in cbiokit); the paper uses
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

cmp.summary      # (group, score): n, mean, median, auc and p against the reference (G, RPT, MPS scales)
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


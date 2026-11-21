[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Gene list scores and background variables (`score_genesets`, `load_covariates`)

Any gene list (PAM50, EMT, hypoxia, a bundled Hallmark / Reactome collection, ...) can be scored per sample with aREA, the machinery of the protein
activity (a gene list is a regulon with all modes of regulation 1; needs R). The scores can adjust the activity analysis or be used on their own:

```python
scores = pr.score_genesets(expression, {"EMT": emt_genes, "Hypoxia": hypoxia_genes})       # samples x lists (NES)
hallmarks = pr.score_genesets(expression, "h")                                             # a bundled collection by name; also one bare gene list -> "score"
```

`method="mad"` (default) takes every gene relative to the median and MAD across the samples of the cohort, so scores say *higher or lower than in
the rest of this cohort* and are not comparable between cohorts. **`method="none"` scores every sample from its own gene ranks** (aREA ranks the genes
within a sample: a robust, ssGSEA-like score that does not depend on the other samples and is comparable between samples and cohorts; tested); the signature is ranked over all genes of the expression. A list is not a subtype
classifier (PAM50 genes give a signature score, not the PAM50 call). `df.attrs["n_genes"]` holds the genes found per list; lists with fewer than
`min_genes` found are left out with a warning.

**Background variables** (covariates of samples) are kept in a small registry, like the gene sets. The tables bundled with the package are in
`pyrea/data/covariates` (now: the TCGA **ABSOLUTE** purity and ploidy of the PanCanAtlas, about 65 KB); tables you add go to
`$PYREA_DATA/covariates` (default `~/.pyrea/covariates`) and are used alike:

```python
pr.available_covariates()                                   # variable, description, table, source, location (bundled / user), n_samples
pr.load_covariates("absolute_purity", samples=expr.columns, sample_id=pr.tcga_sample_id)    # rows for your samples (NaN where missing)
pr.add_covariates("estimate", table, description={"estimate_stromal": "..."}, source="Yoshihara 2013, TCGA scores")   # samples x numeric variables
pr.remove_covariates("estimate")                            # tables of your own only
```

Samples are matched by name; for TCGA the registry key is `TCGA-XX-XXXX-01` (`pr.tcga_sample_id` maps `TCGA.2H.A9GF.01A.11R...` to it). Variable
names must be unique across the registry (use a prefix). The ABSOLUTE table keeps rows of every call status (`absolute_called` = 1 for `called`).
`pr.resolve_covariates(spec, expression)` turns all forms of `covariates` (registry names, `{"name": genes}` dicts to score, tables, lists of these)
into one table; `activity_specificity_across_cohorts` calls it for every cohort. **Caution:** gene list scores come from the same expression as the
activities, so a list that contains the regulators or their targets (an EMT list for ZEB1) removes part of what is tested; compare with and without.

## Stroma and immune scores: ESTIMATE (`read_estimate`, `import_estimate`)

The ESTIMATE scores (stromal, immune and their sum; Yoshihara et al., Nat Commun 2013) are **not part of the package**: the source states educational and
research use only and no redistribution is described. Download the scores per disease (RNA-seq V2) from
[bioinformatics.mdanderson.org/estimate](https://bioinformatics.mdanderson.org/estimate/) into a folder (tab-separated files `ID`, `Stromal_score`,
`Immune_score`, `ESTIMATE_score`, named by disease, e.g. `stomach_adenocarcinoma_RNAseqV2.txt`) and read them with:

```python
est = pr.read_estimate("../Shared/data/estimate")                # samples x estimate_stromal / estimate_immune / estimate_score + cohort
est = pr.read_estimate("../Shared/data/estimate", cohorts=["STAD", "BRCA"])
pr.import_estimate("../Shared/data/estimate")                    # once: into your own registry ($PYREA_DATA/covariates), not into the package

ma = pr.activity_specificity_across_cohorts(cohorts, sets, covariates=["absolute_purity", "estimate_stromal", "estimate_immune"],
                                            sample_id=pr.tcga_sample_id)
```

The index is the sample key of the files (`TCGA-XX-XXXX-NN`, the last two digits the sample type: 01 primary, 06 metastasis, 03 blood-derived for LAML), the
same as `pr.tcga_sample_id` produces, so the scores match the expression columns directly (STAD 391 of 391, LAML 173 of 173 samples in the test data).
Colorectal cancer is one file (`COADREAD`). A sample in several files is kept once with a warning. The scores are unit-free, comparable between samples, and the
ESTIMATE score is the sum of the other two (low = high tumour purity). Compared with the gene list scores from `score_genesets`, they come from the published
signatures and are the standard in the literature; `score_genesets(..., method="none")` is the self-computed, rank-based counterpart.

**Example (ZCCHC24 | ZEB1, partial Spearman correlation of the activities, three TCGA cohorts):**

| adjusted for | ESCA | OV | STAD |
| --- | --- | --- | --- |
| nothing | 0.96 | 0.94 | 0.94 |
| `absolute_purity` | 0.96 | 0.92 | 0.91 |
| `estimate_stromal` + `estimate_immune` | 0.81 | 0.77 | 0.76 |
| purity + both | 0.81 | 0.76 | 0.74 |

The purity alone explains little, the ESTIMATE stromal and immune scores explain a good part of the link; the worst partner percentile of the two regulators
stays in the best few percent in all cohorts (0.003-0.034).

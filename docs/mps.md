[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Mutant phenotype score and locus-specific variants

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

### RPT methods

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

### How the relative likelihood is defined (`method`)

`RL(x) = (p_M(x) - p_WT(x)) / (p_M(x) + p_WT(x))` needs a probability `p` of the mutant and the WT phenotype at the
trait value `x`. Two definitions are implemented; both start from Gaussian kernel estimates (Scott's bandwidth) of the
mutant and WT samples:

| `method` | `p_M`, `p_WT` | behaviour |
| --- | --- | --- |
| `"density"` (default) | the kernel densities at `x` | `RL = (LR - 1) / (LR + 1)` with `LR = p_M / p_WT`; local, can be non-monotone (bimodal mutants), shows plateaus, steps and saturation at +-1 |
| `"cdf"` | cumulative distribution functions: after orienting the trait so that the mutants are the higher group, `p_M = F_M(x)` (mutants at or below `x`) and `p_WT = 1 - F_WT(x)` (WT at or above `x`) | RL is monotone in the trait and runs smoothly from -1 to +1 (sigmoid rank plot); exactly antisymmetric when the groups are swapped; more sensitive and less specific |

**What `RL > 0.5` means depends on the method.**

- `"density"`: `RL > 0.5` is `LR > 3`, i.e. at this trait value the sample is **three times as likely to come from the mutant as
  from the WT distribution** (`RL = 2 P(M | x) - 1` for equal class priors, so `P(M | x) > 0.75`). This is the reading used by
  Alvarez et al. for the MPS-defined mutant and WT phenotypes: in the Results on Fig. 5a, samples with MPS >= 0.5 are described
  as "threefold more likely to have mutated activity" and those with MPS <= -0.5 as "threefold more likely to have WT activity".
- `"cdf"`: `RL > 0.5` means `F_M(x) / (1 - F_WT(x)) > 3`, a ratio of *tail probabilities* (mutants at or below `x` against WT at or above
  `x`). **The likelihood-ratio reading does not carry over**: at the threshold the density ratio is smaller than 3. For
  normal groups (unit variance, equal priors) whose means are 1, 2 or 3 SD apart:

  | mean difference | `x` where `RL = 0.5` (`density` / `cdf`) | density ratio there (`density` / `cdf`) | `P(M \| x)` there (`density` / `cdf`) |
  | --- | --- | --- | --- |
  | 1 SD | 1.60 / 0.98 | 3.00 / 1.61 | 0.75 / 0.62 |
  | 2 SD | 1.55 / 1.36 | 3.00 / 2.05 | 0.75 / 0.67 |
  | 3 SD | 1.87 / 1.78 | 3.00 / 2.34 | 0.75 / 0.70 |

  The `cdf` threshold lies lower on the trait, so more WT samples reach `RL > 0.5` (1 SD: 16 % instead of 5 % of the WT samples;
  24 % instead of 17 % of the calls are WT). The difference shrinks for well separated groups. Do not describe `cdf` values as
  "three times as likely"; say that the MPS is a ratio of tail probabilities (`RL > 0.5`: ratio above 3).

The Methods of [Alvarez et al. 2016](https://doi.org/10.1038/ng.3593) say that the densities are estimated with a Gaussian
kernel and that "the probabilities, computed by the derived cumulative distribution functions" enter the relative
likelihood, but give no formula (the supplement shows density histograms and rank plots and has none either). `"cdf"`
follows the construction that Mariano Alvarez uses for the mode of regulation in the `viper` package (`viper::TFscore`:
signed ratios of `pnorm` tail probabilities of the mixture components). **It is an interpretation, not a verified
reproduction**; state the `method` when reporting results, and consider `"density"` as a sensitivity analysis (or the
primary analysis when the likelihood-ratio statement is needed).

What changes with the definition (TCGA, `RL > 0.5`): EGFR in GBM, mutated 68 % vs. WT 11 % (`"density"`) and 67 % vs. 21 %
(`"cdf"`); ERBB2 amplification in STAD, 55 % vs. 5 % and 86 % vs. 13 %. The ranking of samples within one trait is
the same, so the AUC of a single trait does not change; what changes is the calibration of the threshold. Two
consequences:

- With `"cdf"` any value beyond the mutant range scores as mutant-like (RL -> 1), because the tail probabilities are
  cumulative. A group that is *stronger* than the mutants therefore looks like them (see the asymmetry in the
  phenocopy section); with `"density"` such a group lies outside the mutant density and scores low.
- `bw_adjust` mainly matters for `"density"`; for `"cdf"` the CDFs are already smooth.

### Smoother or more transparent rank plots

The rank-sorted MPS is a sigmoid for a gene with a clear, one-sided effect (EGFR in GBM, Alvarez et al., Fig. 5). With
`method="cdf"` it is smooth in the single traits for every gene; with `method="density"` (default) it can show plateaus
and steps. What remains in both:

- *saturation* (`"density"`): with Scott's bandwidth the RL reaches +-1 in the tails of the densities (about 5 % of the
  samples for ERBB2 / EGFR). `bw_adjust` multiplies the kernel bandwidth (`bw_adjust=1.5-2` removes the saturation);
  larger values flatten the curve and lower the share of samples that reach RL > 0.5. It is accepted by
  `relative_likelihood`, `mutant_phenotype_score`, `locus_specific_mps`, `mps_targets`, `compare_phenotypes` and
  `plot_rl_diagnostic`; the default 1.0 is Scott's rule.
- *steps between G and RPT*: the MPS takes the larger |RL| of the two traits, so it can jump where they disagree.
  `plot_mps_traits` draws the RL of G, of RPT and the combined MPS side by side, so the trait that causes a jump is
  visible.

```python
mps = pr.mutant_phenotype_score(mutations, vpres, rpt)                          # method="density" (LR > 3)
mps_c = pr.mutant_phenotype_score(mutations, vpres, rpt, method="cdf")          # tail probabilities, smoother
mps_s = pr.mutant_phenotype_score(mutations, vpres, rpt, bw_adjust=1.5)         # density, smoothed kernel
fig = pr.plot_mps_traits("EGFR", mutations, vpres, rpt)                          # panels: G, RPT, MPS
```

Rows that are not genes (e.g. a gene fusion) are scored on the activity of one or more chosen
proteins:

```python
summary, mps = pr.mps_targets(fusion_matrix, ["RHOA", "PTK2", "YAP1"], vpres, rpt, expression=emat)
```

## Individual variants (locus-specific)

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

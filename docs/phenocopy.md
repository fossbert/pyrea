[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Do two alterations act alike? (phenocopy)

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

## How to read `cmp.cross` (the phenocopy test)

Question: **does alteration B put the protein into the same state as alteration A?** (A, B: e.g. an activating
mutation and an amplification, a hotspot and a rare variant, a known driver and a gene fusion.)

1. *Scale of A.* From the activity of the target in the A samples and in the reference (samples with none of the
   alterations) the relative likelihood `RL(x) = (p_A(x) - p_ref(x)) / (p_A(x) + p_ref(x))` is estimated for every
   activity value (G and RPT, the one with the larger |RL| counts; `p` from the CDFs or the densities, see `method`). RL > 0 means: more like A than like the
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

**The asymmetry depends on `method`.** With `method="cdf"` the scale is monotone: every sample beyond the range
of A scores as A-like, so a stronger group (e.g. an amplification against a mutation) scores high on A's scale by
construction, while A (weaker) scores lower on the scale of the stronger group. With `method="density"` a group outside A's
density scores low, which can also be an overshoot and not a different state. Example, ERBB2 in STAD (amplification vs.
mutation, `cdf` / `density` (default)): the amplified samples on the mutation scale `frac_mutant_phenotype` 0.71 / 0.00, the
mutated samples on the amplification scale 0.19 / 0.04. Report the `method`, and treat "B high on A's scale" as
"B is at least as extreme as A in the same direction", not as proof of the same state; `cmp.pairwise` (do A and B differ?)
and the raw `cmp.scores` show whether B simply exceeds A.

`cmp.pairwise` answers the complementary question (do A and B differ from each other?). A result that looks
like a phenocopy in `cross` and shows no difference in `pairwise` is the strongest case.

## Where the AUC comes from

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

- The RL is a value per sample derived from the kernel estimates of the two groups. With `method="density"` it is local and can
  be non-monotone (a gene whose mutated samples are bimodal); with `method="cdf"` it is a monotone function of the trait. The AUC
  is *one number per comparison* and asks only whether the group is shifted up or down as a whole. A bimodal group with one mode
  on each side of the reference gives an AUC near 0.5 although the distributions differ; the density RL (and
  `frac_mutant_phenotype`) would show it. Look at both.
- On G or RPT alone the AUC does not depend on `bw_adjust` or `method`. On the MPS scale it does, because the scale is built from
  the kernel estimates and takes the larger |RL| of G and RPT (not a monotone function of either).
- With few samples the AUC is uncertain, so every table has a bootstrap interval, `auc_low` / `auc_high` (95 %,
  `alpha=0.05`). Each group and the reference are resampled separately (`n_boot=1000`, `seed=0`; `n_boot=0` switches it off
  and gives NaN). For the MPS scales the scale itself is estimated again in every resample from the resampled group and
  reference, so the uncertainty of the scale is included and the interval is not too narrow. The `p` is the Mann-Whitney
  test of the original data, not corrected for multiple testing. Runtime: about 10-25 s for ~500 samples and two
  groups with 1000 resamples (more groups = one more MPS scale each; use `n_boot=200` while exploring).
  For small groups (a rare fusion, n < 15) also show the individual points next to the AUC (`cmp.scores`).

```python
cmp = pr.compare_phenotypes(groups, "RHOA", vpres, rpt)            # intervals included
cmp.cross[["n", "frac_mutant_phenotype", "auc", "auc_low", "auc_high", "p"]]
```

## Illustrating a phenocopy

`compare_phenotypes` returns a `PhenotypeComparison` that carries its settings (target, `method`, reference, groups, `lr`,
`alpha`, `n_boot`) and draws the figures itself (matplotlib only, no cbrviz needed):

```python
cmp = pr.compare_phenotypes(groups, "RHOA", vpres, rpt)
fig = cmp.plot()                                   # four panels A-D, see below
cmp.plot_scores("G")                               # A: one panel, e.g. on an own axes:  cmp.plot_scores("RPT", ax=ax)
cmp.plot_scale_scatter("RHOA mut", "CLDN-ARHGAP fusion")   # B
cmp.plot_cross()                                   # C
cmp.plot_auc(pairwise=True)                        # D
```

| panel | shows | reading |
| --- | --- | --- |
| A `plot_scores` | the score (`G`, `RPT` or `MPS[<group>]`) per group, one point per sample, boxplot behind | is B shifted in the same direction as A? With few samples the points matter more than the box |
| B `plot_scale_scatter` | every sample on the MPS scale of A (x) and of B (y); dotted lines: mutant / WT phenotype thresholds and the diagonal | phenocopy puts the B samples with the A samples in the upper right; a group that only lies further out along the diagonal is stronger, not necessarily the same |
| C `plot_cross` | matrix of `cmp.cross`: AUC against the reference with its interval and the share with the mutant phenotype; dashed frame = in-sample | phenocopy: the off-diagonal AUC is above 0.5 with an interval clear of 0.5, in both directions |
| D `plot_auc` | forest plot of the AUCs with their bootstrap intervals (filled: the group was not used to build the scale; hollow: in-sample; grey squares: A against B) | how wide the intervals are, which statements they carry, and whether A and B differ from each other |

Colours and group labels (with `n`) are the same in all panels; the title names the target and the RL definition (`method`).

The panels are clean (no numbers, no brackets), so nothing overloads a figure that does not need statistics. Each is drawn on a
matplotlib `Axes` that you can pass (`ax=`) and that is returned, so axes, limits, labels and further artists can be changed
afterwards (`cmp.plot(axes=...)` takes 4 axes, or 2 with a single group, and returns them). Statistics are added afterwards and
selectively, as with `cbrviz.StripBox.add_pair_p` (`ax=None` uses the current axes; `**text_kwargs` go to matplotlib):

| panel | add statistics | writes |
| --- | --- | --- |
| A `plot_scores` | `cmp.add_group_stats(score, ax=ax)` | AUC with its bootstrap interval (`ci=False`: without) and the Mann-Whitney p of every group against the reference, above each group |
| A | `cmp.add_pair_stats(a, b, score, ax=ax)` | bracket between two groups with the AUC and p of their comparison (`cmp.pairwise`); `yoffset` raises it above the group texts, `auc=False` prints p only |
| B `plot_scale_scatter` | `cmp.add_scatter_stats(ax=ax)` | AUC of each group on both scales in a small text box (`loc=`) |
| C `plot_cross` | `cmp.add_cross_stats(ax=ax)` | AUC, interval and share with the mutant phenotype in the cells |
| D `plot_auc` | `cmp.add_auc_p(ax=ax)` | p of every row at the right edge |

**Selecting what is shown.** Scores are named `"G"`, `"RPT"` (if given) and `"MPS[<group>]"` (`cmp.score_names`). `plot_scores(score)`
shows one score; `plot_auc(scores=[...])` shows the chosen scores as rows, in the order given; `groups=[...]` restricts
the groups (and their order) in `plot_scores`, `plot_scale_scatter`, `plot_cross` and `plot_auc` (`reference=False` hides the
reference in `plot_scores`); `plot_scale_scatter(scale_a, scale_b)` chooses the two scales. `cmp.plot(scores=..., groups=...)`
passes the choice on to the four panels, and the `add_*` functions follow what the panel shows.

```python
cmp.plot_auc(scores=["G", "MPS[RHOA mut]"], groups=["CLDN-ARHGAP fusion"])      # one group, two scores
cmp.plot(scores=["G"], stats=True)                                              # overview with a forest plot of G only
```

`exact_p=True` prints p as `1.2e-03`. `cmp.plot(stats=True)` adds all of them, for a figure that goes into a manuscript.
`colors={"group": "colour"}` sets group colours consistently in the panels that show groups.

```python
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), gridspec_kw={"width_ratios": [1.2, 1.2, 1.5]})
pal = {"neither": "0.7", "RHOA mut": "tab:red", "CLDN-ARHGAP fusion": "tab:blue"}
cmp.plot_scores("G", ax=axes[0], colors=pal)
cmp.add_group_stats("G", ax=axes[0], ci=False)                          # only where a number is needed
cmp.add_pair_stats("RHOA mut", "CLDN-ARHGAP fusion", "G", ax=axes[0], yoffset=0.25)
cmp.plot_scores("MPS[RHOA mut]", ax=axes[1], colors=pal)
axes[1].axhline(0.5, ls=":", lw=0.5, color="k")                         # own additions: it is an ordinary Axes
cmp.plot_auc(ax=axes[2], scores=["G", "RPT", "MPS[RHOA mut]"], colors=pal)
cmp.add_auc_p(ax=axes[2])
```

For a custom figure, `cmp.scale_xy(a, b)` returns the data of panel B and `pyrea.phenotypes_plot.group_colors(cmp)` the
colours; with [cbrviz](https://github.com/fossbert/cbrviz):

```python
from cbrviz import XYview, StripBox
from pyrea.phenotypes_plot import group_colors

cols = group_colors(cmp)
order = [cmp.reference] + list(cmp.groups)
xy, grp = cmp.scale_xy("RHOA mut", "CLDN-ARHGAP fusion")          # columns scale_A, scale_B; overlap / unprofiled dropped
cats = [g for g in order if g in set(grp)]
xv = XYview(xy, s1=grp, s1_order=cats, s1_colors=[cols[g] for g in cats], s=14, alpha=0.8)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8))
ax1.scatter(**xv.scatter_call); xv.add_xy_line(ax=ax1); ax1.legend(handles=xv.handles, frameon=False, fontsize=6)
df = cmp.scores.loc[cmp.scores["group"].isin(order), ["G", "group"]]
sb = StripBox(df, s1_order=order, s1_colors=[cols[g] for g in order]); sb.boxplt(ax=ax2, widths=0.6); sb.add_strips(ax=ax2, s=8)
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

# Does the effect differ between subtypes? (stratified comparison)

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

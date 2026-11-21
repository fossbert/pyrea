[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Does a co-mutation modify the effect of a driver? (`comutation_effect`)

Question: does a second alteration C (e.g. ARID1A, TP53) change the activity of a protein in carriers of a driver D (e.g. RHOA)
beyond what it does on its own? Design: a 2 x 2 table of D and C (neither / D only / C only / D+C, see `cbk.joint_status`), the
activity `y` of the target compared across the cells.

- *Main effect of the partner*: C only vs. neither (`C within not-D`); *simple effect with the driver*: D+C vs. D only
  (`C within D`); **interaction** = the difference of the two, i.e. whether C changes what D does. An effect of C that is the same with
  and without D is a main effect, not a co-mutation effect.
- **Linear model** `y = a + b_D D + b_C C + b_DC D C + g'Z + e` with covariates `Z` (e.g. subtype, cohort; categorical columns are
  dummy-coded, redundant columns such as subtypes nested in cohorts are dropped with a warning), HC3-robust standard errors and
  Freedman-Lane permutation p-values (valid for small cells). `b_DC` is the interaction on the additive activity scale.
- **Rank-based effects**: Cliff's delta (`2 * AUC - 1`) for each simple effect and both interactions (`interaction C by D`,
  `interaction D by C`), with bootstrap intervals in which the four cells are resampled separately; with covariates computed on values
  adjusted for them. The interaction on the delta scale is the difference of two deltas and depends on the scale: descriptive.
- `min_detectable_interaction`: the smallest interaction (in residual SDs, 80 % power) the cell sizes allow. With cells of 4-8 samples it
  is 1.3-1.7 SD: a non-significant interaction then says "not testable", not "absent".

```python
import cbiokit as cbk, pyrea as pr

mat = cbk.alteration_matrix(ex, types=("MUT",), by="PATIENT_ID")        # NaN = not profiled
cbk.joint_status(mat.loc["RHOA"], mat.loc["ARID1A"]).value_counts()       # cell sizes
res = pr.comutation_effect(mat.loc["RHOA"], mat.loc["ARID1A"], "RHOA", vpres, rpt=rpt, expression=emat,
                           covariates=meta[["cohort", "subtype"]], n_boot=1000, n_perm=3000)
res.cells          # n, mean, median per cell and score (G, RPT, expr)
res.effects        # Cliff's delta with interval, per contrast and score
res.model          # beta, HC3 se / interval, p_hc3, p_perm for D, C and D:C
res               # CoMutationResult(...; min detectable interaction ...)
```

A patient with a deletion of the driver gene is "absent" in a mutation row: set such patients to NaN first
(`mat.loc["RHOA"].where(deletions.loc["RHOA"] != 1)`). For many partners run the function per partner and correct for
multiple testing (BH) yourself; the double-mutant cells are small, so treat results as hypotheses.

The figures follow the design of the phenocopy figures (clean panels, an `Axes` that can be passed, statistics added afterwards):

| panel | draw | add statistics |
| --- | --- | --- |
| A score per cell | `res.plot_cells(score)` | `res.add_cell_stats(a, b, score, ax=ax)`: bracket with delta and Mann-Whitney p |
| B interaction plot | `res.plot_interaction(score)` | `res.add_interaction_stats(score, ax=ax)`: `beta` of `D:C` with interval, p(perm), p(HC3), detectable effect |
| C forest plot | `res.plot_effects(scores=[...], contrasts=[...])` | `res.add_effect_p(ax=ax)` |

`res.plot()` draws all three (`res.plot(stats=True)` with all statistics; `axes=` takes three own axes; `scores=` the rows of the forest plot,
default the plotted score). Parallel lines in B mean no interaction.

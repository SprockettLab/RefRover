# PLAN.md — RefRover research roadmap & restart spec

> Status: **Tier-2 benchmark3 COMPLETE (2026-07-06), 119/120 cells — see §12 for
> the full resume point.** Headline: at k≥10 the selection method barely moves MAG
> yield, the Tier-1 proxies do *not* predict Tier-2 yield, and containment's 3×
> `frac_variance` advantage does *not* translate into MAGs. **`taxonomy_stratified`
> was NOT in this run** (no GTDB matrix was passed). Next experiment: fixed
> k ∈ {10,14,20} + `all_vs_all` ceiling + `taxonomy_stratified`, on the 183-sample
> rewilded-mouse cohort, with a margin-sensitive yield metric. **Start at §12.**
>
> This document supersedes the ad-hoc selector design described in `CLAUDE.md`.
> Where the two disagree, PLAN.md is the source of truth for *what we are building
> next*; CLAUDE.md still describes the pipeline plumbing (sketch → align → coverage
> → format) accurately but its `selectors/` tables are the pre-restart design —
> the live model is the feature-space × rule × k grid (§1).
>
> The existing per-sample selectors (`src/refrover/selectors/`), the containment
> matrix (`containment.py`), and the variance-explained proxy
> (`benchmark.unique_variance_explained`) are the raw material for this restart —
> not throwaway. This plan reorganizes them around a cleaner decomposition and
> points the whole effort at a sharper research question.

---

## 0. The one-sentence goal

For each sample, cheaply pick a small set of reference assemblies to map against
that maximizes the **differential-coverage signal** a binner can exploit — and,
crucially, **learn what about a sample tells us which selection method to use for
it.** The second half is the prize: a per-sample adaptive meta-selector, not a
single global winner.

---

## 1. First-principles framing

Differential-coverage binning separates contigs into genomes using how contig
depth **co-varies across samples**. To get that signal you map each sample's
reads against some reference set and measure per-contig depth. Mapping every
sample against every other sample's assembly is O(n²) and infeasible past a
dozen-ish samples (mapping effort scales quadratically). So for each sample we
**select** a subset of references.

The whole problem decomposes into three independent knobs that the old code
fused together:

```
   FEATURE SPACE   ×   SELECTION RULE   ×   k    →   reference set   →   SCORE
   (what matrix M)     (how to pick)       (budget)    (per sample)     (per sample)
```

- **Feature space** = *what matrix we select on.* This is mostly what people
  mean when they say "the containment method" vs "the Jaccard method" — it's a
  choice of matrix, not a choice of algorithm.
- **Selection rule** = *the algorithm that picks k columns* from that matrix.
- **k** = how many references (fixed for now; adaptive-k is downstream of this).

Separating these turns "compare the methods" into a clean grid instead of a pile
of bespoke selectors.

**Plain English:** imagine a spreadsheet where rows are samples and columns are
candidate reference assemblies, and each cell says "how strongly does this
reference show up in this sample." *Feature space* is how we fill in the cells.
*Selection rule* is how we choose which columns to keep. They're orthogonal — any
rule can run on any spreadsheet.

---

## 2. Terminology (locked)

- **Cohort** = the set of samples in a study.
- **Per-sample assembly** = one sample's own assembly (the microbial mixture in
  that one sample). **Not** a co-assembly of pooled samples. Every sample has
  exactly one assembly, so **sample ↔ assembly is 1:1.**
- "Select k references for sample *i*" therefore means "pick k *other samples*
  whose assemblies sample *i*'s reads will be mapped against."
- The matrices below are all **samples × samples** (rows = samples whose reads we
  map; columns = candidate reference assemblies).

---

## 3. Feature spaces (all three, normalized to one shape)

Every feature space is normalized to the **same `samples × per-sample-assemblies`
matrix shape** so a single scoring function compares them apples-to-apples.
Entry `M[i, j]` = "signal of reference assembly *j* in sample *i*."

| Feature space | `M[i,j]` is… | Needs read sketches? | External DB? | Cost (183-sample mouse set) | Role |
|---|---|---|---|---|---|
| **Jaccard** | Jaccard(assembly_i, assembly_j) | No (assembly sketches only) | No | seconds–minutes | **Floor.** Weakest signal (never sees read abundance); kept to answer "is it even better than random?" |
| **Reads×assembly containment** | fraction of sample *i*'s read k-mers found in assembly *j* | Yes | No | minutes (matrix); read sketching is the one-time cost | **No-DB middle tier.** Same variance idea as GTDB, no database. |
| **GTDB taxa-abundance** | abundance of assembly *j*'s taxa in sample *i* (routed through GTDB; see §3.1) | Yes | **Yes (~3.6 GB)** | **~26 min/sample → ~80 h** + DB | **Best signal**, highest cost & setup burden. |

**Plain English:** Jaccard asks "do these two assemblies look genetically
similar?" — cheap but blind to abundance. Containment asks "do this sample's
reads land in that assembly?" — directly about mapping, no database. GTDB asks
"how abundant is what that assembly represents, in this sample?" using a curated
reference catalog — the richest answer, but you pay for the database and the
compute.

### 3.1 The GTDB → samples×assemblies bridge (the one nontrivial construction)

GTDB gather natively gives **samples × taxa**, but we map reads against
**assemblies**, not against GTDB genomes (this is de novo binning). So we route
the GTDB signal through taxa to land back on assemblies:

```
assembly_j  →  its GTDB taxa  →  abundance of those taxa in sample i  =  M[i, j]
```

**Plain English:** "Assembly *j* is mostly *Bacteroides* and *Akkermansia*. How
much *Bacteroides* + *Akkermansia* is in sample *i*? That number goes in the
cell." This makes GTDB select the same unit (assemblies) as the other two, on the
same scoreboard. It's essentially a taxonomy-mediated, low-rank version of the
containment matrix. This is the one piece of genuinely new construction work.

Both inputs come from `sourmash gather`: the **reads** side (`samples × taxa`,
`run_gtdb_gather.py` → `gtdb_species_matrix.tsv`) and the **assemblies** side
(`assemblies × taxa`, `run_assembly_gather.py` → `gtdb_assembly_matrix.tsv`).
`feature_spaces.gtdb_abundance_matrix(sample_taxa, assembly_taxa)` consumes both;
`load_gtdb_species_matrix()` loads either (identical taxa-major TSV layout). When
`assembly_taxa` is omitted it falls back to the reads matrix as a 1:1 proxy
(M = B·Bᵀ), which runs with no assembly-side gather but conflates "in the sample"
with "in the assembly" — use the real `gtdb_assembly_matrix.tsv` whenever it exists.

> **⚠️ Sub-DB caveat — must validate (do not ship on this alone).** Both gathers
> currently run against a **community sub-database** (`gtdb_subdb/community_species.zip`,
> ~2k sigs) distilled by `build_community_subdb.py` from **5 representative
> samples** gathered against full GTDB, not against the full 143k-sig DB. This is a
> ~60× speedup but it is **circular and dataset-specific**: the 5 reps are chosen
> by the very "which samples represent the community" question this whole project
> exists to answer, and on a cohort *not* dominated by a few well-sampled taxa the
> sub-DB can silently truncate the taxa catalogue — taxa present only in
> non-representative samples never enter the DB, so they vanish from *every*
> sample's profile (reads and assemblies alike). RefRover targets a wide range of
> datasets, many of which 5 samples will not span, so **the sub-DB is a local
> optimization for the mouse set, not part of the method.** Before any sub-DB
> result is trusted: re-run both gathers with `--full-db` and confirm the bridge
> matrix (and the downstream selector ranking) is unchanged within tolerance. See
> §10 open question 6. Until validated, treat sub-DB numbers as provisional.

---

## 4. The unified representation (what makes the grid clean)

**Represent each candidate reference as its column vector over samples.** Then
every geometric rule operates on vectors in the same sample-space, and the
feature space just decides *what's in the vectors*. This dissolves almost all the
"is this combination valid?" questions.

Per-sample selection adds a **candidate filter**: sample *i* only considers
references its reads can plausibly map to (a feature-space-appropriate threshold —
min Jaccard / min containment / min shared-abundance). The rule then runs on
those candidate vectors. So "per-sample" lives in *which columns are eligible*,
and the rule's objective ("span the differential signal") is applied within that
eligible set.

---

## 5. Selection rules + validity grid

Rules operate on candidate reference vectors. ✓ = valid, ✗ = not defined.

| Rule | Jaccard | Reads×asm containment | GTDB abundance | Notes |
|---|:--:|:--:|:--:|---|
| `random` | ✓ | ✓ | ✓ | Baseline. Seeded for reproducibility. |
| `maxmin` | ✓ | ✓ | ✓ | Greedy: maximize min pairwise distance. (current code) |
| `kmedoids` | ✓ | ✓ | ✓ | Cluster medoids. (current code) |
| `archetype` | ✓ | ✓ | ✓ | Convex-hull extremes — span, not centrality. (current code) |
| `greedy_var` | ✓ | ✓ | ✓ | Greedy predicted-variance maximization. (current code) |
| **`css`** (pivoted-QR / CUR) | ✓ | ✓ | ✓ | **NEW.** Picks columns that reconstruct M best; greedily adds the most *new orthogonal* signal each step. |
| **`submodular`** / **`dpp`** | ✓ | ✓ | ✓ | **NEW (defer).** Principled diversity family maxmin approximates. |
| **`taxonomy_stratified`** | ✗ | ✗ | ✓ | **NEW.** One representative per GTDB clade, favoring abundant+variable clades. Needs taxonomy labels. |
| **`all_vs_all`** | ✓ | ✓ | ✓ | **NEW (ceiling, testing-only).** Selects everything; defines the 100%-signal / max-compute reference line. |

**Why `css` is special:** the scoring metric in §6 measures "variance captured by
orthogonal projection onto the selected columns." Pivoted-QR *greedily maximizes
exactly that quantity*, so it's the rule matched to the scoreboard and likely the
upper bound among cheap methods. If a fancier rule can't beat CSS on the index,
that's informative.

**Plain English of `css`:** "Pick the most informative reference. Mentally
subtract everything it already explains from the rest. Now pick the most
informative of what's *left over*. Repeat." It never wastes a pick on a reference
that's redundant with one already chosen.

**Migration note:** the current `ContainmentSelector` is really "maxmin on the
containment feature space." Under this restart it stops being its own class — it's
the `(containment, maxmin)` cell of the grid. The `min_jaccard` parameter
generalizes to a per-feature-space candidate threshold.

---

## 6. Scoring — a *family* of per-sample proxies, not one number

### 6.1 The honest gap
True MAG quality (high / medium / low completeness & contamination) **only exists
downstream**, after binning + CheckM2. We are upstream-only for this round
(§8), so we cannot *measure* MAG yield — we can only build a proxy *shaped like*
it and later test whether the proxy predicts the real thing. We do **not** claim
the upstream index equals MAG yield.

### 6.2 What we want the proxy to resemble
Your target, in your words: *"this method leads to N high-/medium-/low-quality
MAGs, weighting high & medium more heavily."* A binner resolves a genome when that
genome's depth pattern across samples is **distinct** from other genomes'. So:

> "How many MAGs can this reference set yield" ≈ "how many independent,
> high-energy axes of cross-sample variation the selected references buy you."

That's computable upstream from the singular-value spectrum of the selected
columns:

- axes above a **high** energy threshold → "high-quality-resolvable" (weight ↑↑)
- axes above a **medium** threshold → "medium" (weight ↑)
- axes in the noise floor → unresolvable (ignored)

This tiered, weighted count matches the "N high/med MAGs, high worth more"
intuition far better than a single fraction-of-variance number.

### 6.3 The family we compute per (sample, method, k)
We deliberately keep several, because **which upstream proxy best predicts real
MAG yield is itself an open research question** (the Tier-1-predicts-Tier-2
question). Tier 2 later picks the winner.

1. `frac_variance` — orthogonal-projection variance explained
   (the existing `benchmark.unique_variance_explained`). Single number in [0,1].
2. `effective_rank` — participation ratio of the selected columns' spectrum:
   "how many independent axes, softly counted."
3. `tiered_axis_count` *(default / most MAG-shaped)* — the weighted high/medium
   resolvable-axis count from §6.2.

**Plain English:** all three ask "how much *non-redundant* differential signal did
these references buy?" `frac_variance` gives one blended percentage;
`effective_rank` estimates how many distinct signals; `tiered_axis_count` counts
them but pays more for strong, clean ones — closest to "good MAGs."

### 6.4 Per-sample aggregation (locked: option **a**)
With per-sample selection there is no single chosen set — there are N (one per
sample). **Score each sample's own pick, keep the full per-sample distribution,
then average for a headline number.** We do **not** pool all picks into one set.

---

## 7. The primary deliverable: per-sample features → which method wins

**This is the point of the whole exercise, not an add-on.** The mean across
samples hides what we care about. So we keep the full **per-sample × per-method ×
k** score table and ask:

> **What cheaply-measurable property of a sample predicts which selection method
> is best for it?**

If we can answer that, we get an **adaptive meta-selector**: measure a feature of
each incoming sample, then pick the method (and eventually k) best *for that
sample*. This subsumes the adaptive-k goal and is far more compelling than
"method X wins on average by 3%."

### 7.1 Per-sample features to compute
- **species distinctness / separability** — mean pairwise distance among the taxa
  present ("how distinct the species are")
- **diversity & evenness** — richness, Shannon, dominance (one bug or fifty?)
- **n candidate references** passing the mapping threshold
- **saturation-curve shape** — how fast marginal variance decays as k grows
  (already prototyped in `data/mouse_rewilding/saturation_curves.tsv`)

### 7.2 The analysis
Model `winning_method(sample) ~ features`, or
`score(sample, method) − score(sample, baseline) ~ features`. Output is a
human-readable rule, e.g.:

> *"When a sample is dominated by a few distinct species, `css` wins. When it's
> diverse and even, `archetype` wins. When it has very few candidate references,
> everything ties and `random` is fine."*

### 7.3 Overlap analysis (companion)
Per (sample, k), compute the set overlap (Jaccard of chosen reference IDs)
between every pair of methods (extends
`data/mouse_rewilding/selector_agreement_k5.tsv`). The story: **where methods
agree, the choice doesn't matter; where they diverge, the index — and ultimately
the per-sample features — arbitrate.** Divergence is where the science is.

---

## 8. Two-tier evaluation

- **Tier 1 — upstream, alignment-free (COMPLETE on AMY1).** The
  `(feature × rule × k)` grid, per-sample score family (§6), and overlap are
  implemented and have been run on the AMY1 dataset (63 samples, 2 feature spaces,
  6 rules, k ∈ {3,5,8,10}). See §11 for empirical findings.
- **Tier 2 — downstream, gold standard (COMPLETE 2026-07-06, 119/120 cells,
  `work/benchmark3/` on `cbsupoole`). Results + interpretation in §12 — read that
  first. The description below is the run's design.**
  Two prior runs were killed: the first (SLURM array, sequential alignments)
  was too slow; the second (GNU parallel -j8, k ∈ {5,10,15,20,adaptive} × 30
  focals = 900 cells) filled `/local` in 23 hours (SAM intermediates + retained
  BAMs). Two fixes were applied to `mag_benchmark.py`:
  1. BWA→samtools pipe (no SAM on disk)
  2. BAM directory deleted immediately after `run_coverm()` completes
  Current minimal run config:
  - **10 focal assemblies** (`head -10 work/focals_30.txt`; see `work/focals_10.txt`)
  - **k ∈ {10, adaptive}** — fewest cells that still test the MaxMin paradox at
    a k where rules diverge (see §11.6) and characterize the adaptive estimator
  - **Rules: random, maxmin, css** × 2 matrices (Jaccard + containment) = 6
    cells per focal; **60 cells total**
  - **Execution: GNU parallel -j6** on `cbsupoole`, 10 threads per process
    (6 × 10 = 60 cores, leaving 4 headroom), joblog in
    `work/parallel_benchmark3.log`
  - **Output:** `work/benchmark3/`
  - **Estimated completion:** ~25 hours (~2.5 hrs/cell ÷ 6 parallel)
  - **Adaptive k:** `k="adaptive"` sentinel triggers `estimate_k()` per focal
    (containment_saturation method, k_min=3, k_max=20); `k_actual` recorded in
    result.json

  Primary yield metric: `sum_qs` (Σ max(0, completeness − 5×contamination),
  Olm et al. 2017 ISME J); `weighted_mags` kept for comparison. The central
  open question: **does any Tier-1 proxy predict the Tier-2 ranking?**

  **When results are ready:**
  ```bash
  # On cbsupoole:
  conda run -n refrover-benchmark \
      refrover aggregate-results --outdir work/benchmark3 \
      > work/benchmark3/all_results.tsv

  # Rsync to local:
  rsync -avz cbsupoole:/workdir/Sprockett/Projects/CU15_AMY1_Copy_Number/RefRover/work/benchmark3/all_results.tsv \
      ~/Downloads/benchmark3_results.tsv
  ```
  Key plots to make in R:
  - `sum_qs` vs `frac_variance` per cell (does H_frac hold?)
  - `sum_qs` vs `tiered_axis_count` per cell (does H_tier hold?)
  - k-curve per rule (where does MAG yield saturate vs proxy saturation?)
  - adaptive k distribution: histogram of `k_actual` across 10 focals
  - Jaccard vs containment cells: does the 3× variance gap translate to MAGs?
  - MAG yield vs `k_actual` vs fixed k=10 (is adaptive competitive?)

  **If results look clean, expand to:** all 30 focals × k ∈ {5,10,15,20,adaptive}
  (the original benchmark3 extended config) to confirm saturation elbow and
  improve statistical power.

---

## 11. AMY1 empirical Tier-1 findings (2026-06)

Dataset: 63 samples (AMY1 cohort, BRC BioHPC `cbsupoole`). Both feature spaces
built from sourmash k=31 scaled=1000 sketches. `refrover rank-selectors` run with
6 rules × k ∈ {3,5,8,10} × 2 matrices.

### 11.1 Containment matrix explains ~3× more variance than Jaccard

At k=10, the best rule (css) achieves frac_variance = **0.685** on containment
vs **0.248** on Jaccard. This is consistent with containment directly measuring
read-mapping affinity while Jaccard measures assembly-level k-mer overlap — a
noisier proxy for the same thing. Implication: Tier-2 benchmark cells run on the
containment feature space should produce higher MAG yields. This is an empirical
prediction to verify.

### 11.2 The frac_variance vs tiered_axis_count disagreement (the MaxMin paradox)

The two proxy metrics give opposite rankings for `maxmin`:

| Metric at k=10 | Winner | Loser |
|---|---|---|
| `frac_variance` | css (0.685 / 0.248) | **maxmin last** (0.640 / 0.216) |
| `tiered_axis_count` | **maxmin first** (7.25 / 19.0) | random last (3.97 / 12.6) |

*(values: containment / Jaccard)*

**Why:** MaxMin maximizes pairwise correlation-distance between selected
references — by construction every picked reference is far from every already-
chosen one. This makes each reference individually informative (high
tiered_axis_count) but wastes picks on the *periphery* of the variance
distribution rather than its *core* (low frac_variance). CSS and greedy_var do
the opposite: they greedily grab the highest-variance directions, which may
cluster if the dataset has a dominant axis.

This disagreement is **robust**: it holds on both matrices and at all k values
tested. It is the central empirical question for the Tier-2 benchmark to resolve.
If real MAG yield tracks frac_variance, css/greedy_var will win. If it tracks
tiered_axis_count, maxmin will win despite its low frac_variance.

### 11.3 Rule differences are small at k≥5 on containment

On the containment matrix at k=10, all six non-random rules span only 0.048
absolute in frac_variance (0.637–0.685). The practical consequence: if reads are
available to build the containment matrix, the *choice of rule matters less*.
The bigger lever is the feature space itself, not the algorithm.

### 11.4 CSS is confirmed competitive with greedy_var

At k=10: css frac_variance 0.685 (containment) / 0.248 (Jaccard) vs greedy_var
0.682 / 0.246. The gap is within noise for most datasets. CSS has the theoretical
edge (it directly optimizes frac_variance via pivoted-QR) and is the cleaner
algorithm; treat it as the reference upper bound among cheap rules.

### 11.5 k saturation not yet characterized

At k=10 with containment, frac_variance (0.685) is still clearly rising. The
elbow has not been reached for this 63-sample dataset. Run with k ∈
{3,5,8,10,15,20} to locate the saturation point; this also sets a principled
upper bound for the Tier-2 extended run.

### 11.6 Jaccard Tier-1 full results (all 6 rules × k ∈ {3,5,8,10})

Full run on AMY1 (63 samples, `--min-similarity 0.05`):

| rule | k=3 frac_var | k=5 frac_var | k=8 frac_var | k=10 frac_var | k=10 tiered |
|---|---|---|---|---|---|
| css | 0.099 | 0.146 | 0.209 | **0.248** | 18.2 |
| greedy_var | 0.093 | 0.142 | 0.205 | 0.246 | 18.7 |
| kmedoids | 0.085 | 0.147 | 0.209 | 0.244 | 15.6 |
| archetype | 0.080 | 0.139 | 0.204 | 0.241 | 15.1 |
| random | 0.090 | 0.134 | 0.192 | 0.229 | 12.6 |
| maxmin | 0.079 | 0.121 | 0.180 | 0.216 | **19.0** |

Key observations:
- **k=3: complete rule indistinguishability.** `tiered_axis_count` = 6.0 for
  every rule except random (5.9) and archetype (5.98) — at the theoretical max
  of 2×3. No information about rule quality at this k.
- **k=5: near-indistinguishability.** `tiered_axis_count` span = 9.67–9.95
  (3%). frac_variance span = 0.134–0.147 (10% relative). Not enough signal to
  pick a rule. Decision: **drop k=3 from Tier-2; keep k=5 as minimum baseline.**
- **k=10: MaxMin paradox confirmed on Jaccard** (matches containment, §11.2):
  maxmin last on frac_variance (0.216), first on tiered_axis_count (19.0).
- **Jaccard variance low throughout:** even at k=10, best frac_variance = 0.248
  (css). Containment reaches 0.685. This 3× gap makes containment the more
  informative feature space for Tier-2 cells; both are included to test whether
  that gap translates to real MAG yield.

---

## 9. Build order (Tier 1)

1. **Feature-space builders** → three functions returning the uniform
   `samples × assemblies` DataFrame:
   - `jaccard_matrix` (from assembly sketches — partly exists via `sketch.py`)
   - `containment_matrix` (exists in `containment.py`)
   - `gtdb_abundance_matrix` (NEW — the §3.1 taxa-routing bridge)
2. **Rule interface refactor** — references-as-column-vectors + per-feature-space
   candidate filter. Adapt the five existing rules; collapse `ContainmentSelector`
   into the grid. Add `css`. Add `all_vs_all` and `taxonomy_stratified`. Defer
   `submodular`/`dpp`.
3. **Score family** (§6) — generalize `benchmark.unique_variance_explained` into a
   `scores` module returning `frac_variance`, `effective_rank`, `tiered_axis_count`.
4. **Per-sample feature computation** (§7.1).
5. **Grid runner** — produce the tidy `sample × feature_space × rule × k →
   {scores}` table + the overlap table.
6. **Analysis & report** — distributions, feature→method-choice model, overlap,
   run on `data/mouse_rewilding/`.

---

## 10. Open questions (carried forward)

1. **Which Tier-1 proxy predicts Tier-2 MAG yield?** (§6.3, §8, §11.2) — the
   central bet, now sharpened. The Tier-1 run revealed a clean disagreement between
   `frac_variance` (css/greedy_var best) and `tiered_axis_count` (maxmin best).
   The Tier-2 benchmark directly tests which is the better predictor. Hypotheses:
   - *H_frac:* frac_variance predicts MAG yield because differential coverage
     binning benefits from capturing as much cross-sample variation as possible.
   - *H_tier:* tiered_axis_count predicts MAG yield because individually
     informative references (each adding a clean, distinct signal) matter more
     than aggregate variance captured.
   If H_frac, css/greedy_var will rank first in Tier-2 despite maxmin's
   tiered_axis_count advantage. If H_tier, maxmin will outperform on real MAG
   yield despite its low frac_variance. A mixed result (different winner per k or
   per focal assembly) points toward the per-sample adaptive meta-selector (§7).
   **MAG yield metric:** both `weighted_mags` (2×n_high + 1×n_medium) and `sum_qs`
   (Σ max(0, completeness − 5×contamination), Olm et al. 2017 ISME J) are
   recorded. `sum_qs` is preferred — continuous, captures within-tier variation,
   5× contamination penalty is the field standard. Retire `weighted_mags` from
   primary analysis if `sum_qs` proves equally or more discriminating.
2. **Adaptive k.** Once the per-sample feature model exists, does the same feature
   set predict the *elbow* of the saturation curve?
3. **Candidate threshold per feature space.** What min-similarity / min-containment
   / min-abundance actually ensures reads map at useful rates? Likely
   dataset-dependent.
4. **Does per-sample selection beat the current global prototype set**, and on
   *which* samples (the heterogeneous-dataset hypothesis)?
5. **Feedback selector** (map → measure → iterate) — still the most accurate, most
   expensive; revisit only if Tier-1 proxies prove unreliable.
6. **GTDB sub-DB vs full-DB (validation, blocking before any GTDB claim).** Does
   the 5-rep community sub-database (§3.1 caveat) recover the same taxa profiles —
   and the same selector ranking — as the full 143k-sig GTDB? Run both gathers
   with `--full-db` and diff. The deeper question is whether a sub-DB built from
   *any* small sample subset is sound on heterogeneous cohorts, or whether the
   full DB (or a non-circular catalogue construction) is required in general.
   **HPC note:** the full-DB run is ~26 min/sample → ~80 h serial for 182
   assemblies, so it must be parallelized. The per-sample gather is embarrassingly
   parallel and `run_assembly_gather.py` decouples aggregation (`--aggregate-only`)
   from gather, but it is **not yet HPC-ready**: there is no shard/stride flag to
   assign disjoint samples to array tasks (only a `--n-samples` prefix), so N
   concurrent instances would race on the same outputs. Needs a `--shard i/N`
   (or explicit sample-list) option + a Slurm/array-job wrapper before the
   full-DB validation can run on a cluster.

---

## 12. Current status & resume point (2026-07-06)

**Read this section first when returning to the project.** Everything below §11
predates the Tier-2 benchmark3 results.

### 12.1 benchmark3 (Tier-2) is complete — 119/120 cells

Run as executed on `cbsupoole` (`work/benchmark3/`):
- 2 feature spaces (jaccard, containment) × 3 rules (**random, maxmin, css**) ×
  k ∈ {10, adaptive} × 10 focal assemblies = **120 cells**
- GNU `parallel -j6`, 10 threads/cell; `sum_qs` primary metric
- One cell failed (jaccard/random/adaptive/336TP1, MetaBAT2 `bad_lexical_cast`
  from a NaN depth → empty field; root-caused and fixed, see §12.5). It does not
  affect any conclusion; closing it is optional.

Aggregate + analyze (the analyzer is new — §12.5):
```bash
conda run -n refrover-benchmark refrover aggregate-results --outdir work/benchmark3
python scripts/analyze_benchmark.py \
    --results work/benchmark3/benchmark_results.tsv \
    --outdir  work/benchmark3/analysis/
```

### 12.2 Findings (primary metric `sum_qs`)

1. **Selection method barely moves yield.** All 12 (matrix, rule, k) configs
   cluster at **2450–2560 `sum_qs`** (~4.5% spread) with per-config std ~800
   (SEM ~250 over 10 focals) → configs are **statistically indistinguishable**.
   Focal-assembly identity drives ~30% of the variance; matrix/rule/k ~4%.
2. **Tier-1 proxies do NOT predict Tier-2 yield.** Within-(matrix, focal)
   Spearman ≈ 0 for all three: `frac_variance` −0.002, `effective_rank` −0.081,
   `tiered_axis_count` −0.022. Top-1 hit rate (proxy picks the real yield winner)
   20–30% for frac_variance vs **~17% chance** (6 configs/focal); the other two
   at/below chance. **The Tier-1-predicts-Tier-2 bet fails on this dataset.**
3. **Containment's 3× `frac_variance` advantage does NOT translate to MAGs.**
   Jaccard-vs-containment yield Spearman **0.96–0.99**; mean-yield gap <1%
   (random is even slightly *worse* on containment). `frac_variance` is a
   within-matrix quantity, not comparable across matrices — Tier-2 confirms it.
4. **random ≈ css ≈ maxmin.** No selector beats random by a meaningful margin at
   k≥10. The MaxMin paradox (§11.2) is moot at Tier-2: neither proxy's preferred
   rule is vindicated because nothing separates from random.
5. **Adaptive-k is broken.** It pinned to **k_max=20 for every focal** while real
   yield saturated by **k=10** (k=20 ≈ k=10, sometimes worse). The
   `containment_saturation` elbow overshoots the real yield-saturation point —
   the Tier-1≠Tier-2 gap one level down. See §12.6.

**Reframe (DS, 2026-07-06):** rules converging on similar picks is **not** a
failure — it means the problem is robust and the decision space is small. The
research question was never "which of my rules wins"; it is the deployment
question in §12.3.

### 12.3 The actual deliverable + the ONE experiment that answers it

Deployment question: **on a 100+-sample cohort, how much MAG quality is gained at
k=10 vs 14 vs 20, and does principled sample selection beat naive?** Three ratios
are the entire deliverable:
- `yield(k=20) / yield(k=10)` — is 10 enough, or is 20 worth the compute?
- `yield(best rule) / yield(random)` at each k — does *which* samples matter?
- `yield(k=10) / yield(all_vs_all)` — how much is lost by subsetting at all?

**Next run (single grid):**
- **containment** feature space + **fixed k ∈ {10, 14, 20}** (drop adaptive) +
  **`all_vs_all`** as the yield ceiling
- **rules: random, css, taxonomy_stratified** (naive floor, best geometric, the
  intuitive taxonomic one)
- **dataset: the 183-sample rewilded-mouse cohort** — a real 100+-sample set;
  AMY1's 63 samples cannot answer a 100+-sample question
- **GTDB:** build clade labels once via `sourmash gather` against the **full GTDB
  rs214** (k=31, scaled=1000) — a public release, **not** a sub-DB distilled from
  these samples (avoids the circularity flagged in §10.6 / memory)

### 12.4 Why `taxonomy_stratified` was NOT in benchmark3 (the gap)

It is **fully implemented** — `rules.TaxonomyStratifiedRule` (one high-variance
representative per GTDB clade, ranked by mean×variance), the whole profiling path
in `gtdb.py`, and CLI plumbing (`--gtdb-matrix`, `--gtdb-db`,
`--gtdb-clade-level`, auto-build from read sketches). benchmark3 simply passed
`--rules random,maxmin,css` and **no `--gtdb-matrix`**, so no clade labels
existed and the rule was skipped by the validity check (`grid._rule_valid_for`).
To include it next time:
```bash
refrover gtdb-gather --read-sketches read_sigs/ --gtdb-db /path/gtdb-rs214-reps.k31.zip \
    --outdir work/gtdb/            # shardable: --shard i/N
refrover gtdb-matrix  --gather-dir work/gtdb/gather --outdir work/gtdb/
refrover benchmark ... --gtdb-matrix work/gtdb/gtdb_matrix.tsv \
    --rules random,css,taxonomy_stratified --k-range 10,14,20
```
This is a **priority for the next run.**

### 12.5 Code changes this session (branch `restart-feature-rule-grid`, all pushed)

- **Fix** `cli.aggregate_results_cmd`: missing `import pandas` (NameError crashed
  every `aggregate-results`).
- **Fix** `mag_benchmark.aggregate_results`: skip a stale `error.json` when a
  sibling `result.json` exists (cells that failed then succeeded on `--resume`
  were double-counted as both success and failure).
- **Fix** `formatters/metabat2.write_metabat2`: coerce NaN depth/variance → 0.0
  (empty TSV field was the 336TP1 `bad_lexical_cast` MetaBAT2 crash); warns on fill.
- **Harden** `checkm2.run_checkm2`: fail fast with an actionable message when the
  binary can't be resolved (a raw `Errno 2` was silently sinking cells under
  `parallel` when `$CHECKM2` didn't propagate).
- **Harden** `mag_benchmark._bam_complete`: require the BGZF EOF marker, not just a
  sibling `.bai` (a truncated BAM with a stale index crashed CoverM with
  `BamTruncatedRecord`).
- **New** `scripts/analyze_benchmark.py`: dependency-light (pandas/scipy) analyzer
  with the *correct* stats — within-(matrix,focal) Spearman decomposition + top-1
  hit rate (pooled correlation is confounded across matrices/focals). Handles
  `k="adaptive"` via `k_actual`, which `analyze_benchmark.R` drops to NA.
- Tests: **239 passing** (added checkm2 fail-fast, BAM EOF/truncation, metabat2
  NaN-fill).

### 12.6 Known issues / TODO (carried into next session)

1. **`adaptive_k.estimate_k` pins to k_max.** Diagnose: is the `saturation_fraction`
   (0.1) elbow threshold too strict, or does the residual-variance curve genuinely
   never flatten below k=20 on this data? Only worth fixing **if** the fixed
   k-sweep shows per-sample saturation-k actually *varies* — otherwise hard-code a
   fixed k and retire adaptive-k.
2. **Adaptive-k's value proposition is unproven.** It's a per-sample proxy stopping
   rule; benchmark3 showed its stopping point (20) overshot real yield saturation
   (~10). Validate against the fixed k-sweep before trusting it.
3. **Margin-sensitive metric needed** (newly-resolved MAGs vs. a self/k=1
   baseline) in `mag_quality` + the analyzer. benchmark3's `sum_qs` is dominated
   by the focal's easy MAGs (recovered under any co-map set), which is the main
   reason everything looked flat. Add before the next run or the k-sweep risks
   looking flat for the same insensitivity reason.
4. **CLAUDE.md** still documents the pre-restart `selectors/` design; the live
   model is the feature-space × rule × k grid (§1). Plumbing sections remain
   accurate.
5. One benchmark3 cell (jaccard/random/adaptive/336TP1) unfinished — optional to
   close (`rm depth.txt` then re-run the single focal); changes nothing.

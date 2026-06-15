# PLAN.md — RefRover research roadmap & restart spec

> Status: **active restart (2026-06).** This document supersedes the ad-hoc
> selector design described in `CLAUDE.md`. Where the two disagree, PLAN.md is
> the source of truth for *what we are building next*; CLAUDE.md still describes
> the pipeline plumbing (sketch → align → coverage → format) accurately.
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

- **Tier 1 — upstream, alignment-free (THIS ROUND).** Build the
  `(feature × rule × k)` grid, the per-sample score family (§6), the per-sample
  feature analysis (§7), and overlap. Fast iteration on the mouse data. **No
  binning.**
- **Tier 2 — downstream, gold standard (DEFERRED).** Run the real pipeline +
  MetaBAT2 + CheckM2; count quality-weighted MAGs per CPU-hour. The open question
  Tier 1 sets up: **does any Tier-1 proxy predict the Tier-2 ranking?** If yes,
  selection can be chosen cheaply from Tier 1 alone, forever. `benchmark.run_benchmark`
  is the stub entry point.

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

1. **Which Tier-1 proxy predicts Tier-2 MAG yield?** (§6.3, §8) — the central bet.
   **MAG yield metric:** the benchmark now records both `weighted_mags` (2×n_high +
   1×n_medium, a binary tier scheme) and `sum_qs` (Σ max(0, completeness − 5×contamination),
   the continuous Quality Score from dRep / Olm et al. 2017 ISME J). `sum_qs` is more
   defensible — it captures variation within tiers and penalises contamination
   proportionately. **If `sum_qs` proves a better predictor of the Tier-1 proxy ranking
   or a better discriminator between rules, retire `weighted_mags` from the primary
   analysis.** The 5× contamination penalty is the field standard; revisit only if
   CheckM2 contamination estimates prove systematically noisy on this dataset.
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

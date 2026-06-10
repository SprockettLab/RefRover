# RefRover — Code & Analysis Review

Review date: 2026-06-10. Scope: `src/refrover/` package, `data/mouse_rewilding/` analysis
scripts, and the bridge between them. Test suite passes (69 passed) and the validation
analysis (183-sample GTDB gather + selector comparison) is complete.

The headline finding: **the science has moved ahead of the package.** The validated approach
— cross-sample *containment* with MaxMin selection and a residual-variance saturation curve for
adaptive k — lives entirely in `data/mouse_rewilding/*.py` analysis scripts. The shippable
package still revolves around assembly-vs-assembly *Jaccard* selectors, and the one class that
implements the validated method (`ContainmentSelector`) is unreachable from the CLI or pipeline.
Closing that gap is the single highest-value change.

Suggestions are ordered by priority within each section. File/line references are to the state
reviewed.

---

## 1. Correctness issues (fix first)

### 1.1 MetaBAT2 output has fabricated variance — wrong for differential binning
`coverage.py:38-44` runs `coverm contig --methods mean` only. `formatters/metabat2.py` then
writes the per-sample variance columns as **0** because the variance was never computed. But
MetaBAT2's `jgi_summarize_bam_contig_depths` format *uses* the variance column, and SemiBin2 can
too. Shipping zero-variance defeats the differential signal the whole tool exists to produce.

Fix: request both statistics — `coverm contig --methods mean variance` — and thread the variance
columns through to the formatters. This is a real bug, not a cosmetic one: it silently degrades
every MetaBAT2 run.

### 1.2 Hybrid coverage merging is documented but not implemented
`CLAUDE.md` states hybrid short+long depth merging is "handled in `coverage.py`." It is not —
`run_coverm` runs one CoverM invocation over whatever BAMs exist, with no per-contig
read-count-weighted merge (`coverage.py:38-49`). Either implement the merge or correct the doc.
Right now a hybrid sample silently gets whichever BAMs `align` happened to write, concatenated
without weighting.

### 1.3 Inconsistent "query is always included" contract across selectors
`MaxMinSelector`, `GreedyVarSelector`, `ArchetypeSelector`, and `KMedoidsSelector` all force the
query's own assembly into the result. `RandomSelector` does **not** (`random.py:17-18` —
`rng.sample(candidates, k)` with no query guarantee). The query's own assembly is the one
reference reads are guaranteed to map to, so omitting it skews the baseline comparison and means
"random" isn't a clean control for the others. Either document that random deliberately omits the
query, or make the contract uniform (recommended: a `BaseSelector._ensure_query_first()` helper
that every selector calls).

### 1.4 `ContainmentSelector` assumes assembly ID == sample ID
`containment.py:97` — `own_assembly = query_id  # assumes assembly ID matches sample ID`. The
manifest treats `sample_id` and `assembly` as separate columns (`io.py:4`). For the mouse dataset
they happen to coincide; for any caller where they don't, the wrong column gets pinned first. Pass
the manifest (or an explicit `sample_id → assembly_id` map) into the selector instead of assuming.

### 1.5 ID provenance is fragile end-to-end
Sketch names (`matrix_from_sigs` uses `sig.name or filename stem`, `similarity.py:80`),
sourmash-compare CSV header labels, manifest `sample_id`, and manifest `assembly` are assumed
equal at several boundaries but never reconciled. `load_similarity_matrix` only checks the matrix
is square (`similarity.py:26`), not that its labels match the manifest. A single mislabeled sketch
produces silently wrong assignments. Add one validation point that asserts the similarity/
containment matrix index is a superset of `manifest.sample_id`, and normalize sketch names to
`sample_id` at sketch time.

---

## 2. Performance issues

### 2.1 `ContainmentSelector` recomputes correlations in a Python loop
`containment.py:134-160` calls `np.corrcoef` pairwise inside the greedy loop — O(k · n) corrcoef
calls, each O(samples). For 182 candidates this is the dominant cost and is entirely avoidable.
`compare_selectors.py` already shows the fix: compute the full correlation matrix **once**
(`cont.corr()`), convert to a distance matrix, and index into it. Port that pattern into the
class. Expect ~100× speedup at this scale.

### 2.2 `GreedyVarSelector` uses scalar `.loc` in a triple loop
`greedy_var.py:33-38` — `min(1.0 - float(sim_matrix.loc[c, s]) for s in selected)` inside
`for c in remaining` inside the `while` loop. Pandas scalar `.loc` is ~microseconds each; at
183 samples × k × n_candidates this is seconds per query for no reason. Convert `candidates` to a
NumPy submatrix once and do the min-distance update vectorized (the same `min_dist` array trick
`MaxMinSelector` already uses, `maxmin.py:31-42`). Also: `var_scores` includes the self-term
(`sim_matrix.loc[c, all_ids]` contains `c,c = 1.0`), mildly inflating variance — drop the diagonal.

### 2.3 GTDB gather is serial and memory-bound
`build_community_subdb.py` gathers samples one at a time; the log shows per-sample times swinging
from 13 s to 990 s under memory pressure. A process pool (or GNU parallel / `sourmash multigather`
with a manifest) bounded to a sane worker count would cut wall-clock and smooth the variance. Even
2–3 workers against the 55 MB sub-database would help, since the sub-DB fits in memory many times
over.

---

## 3. Architecture & API consistency

### 3.1 Two selector interfaces, no shared contract
`BaseSelector.select(square_jaccard, query_id) -> list[str]` returns **sample IDs**;
`ContainmentSelector.select(rectangular_containment, query_id) -> list[str]` returns **assembly
IDs** and isn't a `BaseSelector` subclass (`containment.py:44`). The pipeline can't treat them
polymorphically, which is why containment never reached the CLI. Recommended: define one protocol

```python
class Selector(Protocol):
    def select(self, query_id: str, *, matrix: pd.DataFrame, manifest: pd.DataFrame) -> list[str]:
        ...  # returns assembly IDs, always
```

Let both Jaccard and containment selectors implement it; resolve sample→assembly via the manifest
in one place. Returning assembly IDs uniformly also removes the implicit sample==assembly coupling
flagged in 1.4.

### 3.2 Adaptive k works in `select` but not in `run`/pipeline
`cli.py select` supports `--k auto` (`cli.py:90-104`), but `RefRoverPipeline.run()` and `cli.py run`
accept only a fixed integer `k` (`pipeline.py:64-68`, `cli.py:189`). End-to-end runs therefore
can't use the feature the research is about. Move the per-query k estimation into the pipeline's
selection loop and have `cli.py run` accept `--k auto` too.

### 3.3 Unimplemented strategies are exposed as if they work
`SELECTOR_REGISTRY` includes `feedback` (`selectors/__init__.py:15`), which raises
`NotImplementedError`; `benchmark.py:31` is a stub; `cli.py select` advertises `greedy_var` but not
`containment`. A user picking `feedback` gets a crash, not a helpful message. Either gate
unimplemented entries behind a clear "experimental/unimplemented" error at construction, or remove
them from the CLI `Choice` lists until they exist.

### 3.4 `print`/`click.echo` instead of logging
Subprocess wrappers swallow stdout/stderr and re-raise on failure (`coverage.py:45-47`,
`sketch.py`, `align.py`). There's no `--verbose`, no run log, no timing. A `logging`-based setup
with a `--verbose/-v` flag and a per-run `refrover.log` would make the multi-hour pipeline
debuggable without code edits.

---

## 4. Streamline the analysis → converge it with the package

The analysis scripts and the package have forked. Three concrete merges:

### 4.1 Move containment computation into the package
`compute_containment.py` lives only in `data/mouse_rewilding/` but computes the matrix that the
*best-performing* selector depends on. Promote it to `refrover/containment.py` with a
`refrover containment` subcommand (sourmash containment / prefetch under the hood). Then containment
is a first-class signal, not an out-of-tree preprocessing step.

### 4.2 Promote the variance-explained proxy into `benchmark.py`
`compare_selectors.py` computes the orthogonal-projection "unique variance explained" metric — a
fast, **alignment-free** ranking of selectors. This is exactly what `benchmark.py` should do as its
cheap first tier, *before* spending CPU-days on CheckM2. Make `run_benchmark` report both:
(1) the proxy ranking (seconds), and (2) optionally the CheckM2 gold standard (hours). The proxy is
also the only way to benchmark at the 183-sample scale without an aligner.

### 4.3 Port the validated adaptive-k method into `adaptive_k.py`
The package's three estimators (`scree_elbow`, `similarity_gap`, `saturation_curve`) all run on the
**Jaccard** matrix as a proxy. The analysis showed the trustworthy method is greedy **residual
variance** on the **containment** matrix with a 10%-of-first-gain elbow (`compare_selectors.py`:
`saturation_curve` + `elbow_from_gains`), which gave a stable k=4 across all 183 samples. Add it as
a `containment_saturation` method and make it the default when a containment matrix is available.
Keep the Jaccard estimators as fallbacks for the no-containment case.

### 4.4 Have the analysis import the package, not reimplement it
`compare_selectors.py` reimplements `random` and `maxmin` selection inline. If those drift from
`selectors/maxmin.py` the benchmark stops measuring what ships. Import the real classes (once 3.1
gives them a common interface) so the experiment exercises production code.

### 4.5 Minor analysis cleanups
- Consolidate `run_gtdb_gather.py` and `build_community_subdb.py` — they overlap substantially.
- Factor the repeated tidyverse/theme boilerplate in the four `explore_*.R` scripts into a sourced
  `_setup.R`.
- The sub-DB build prints `Sub-database: ... (0 signatures)` (`bd8ykcrxj` log line 15) even on
  success — the count is read from the wrong place. Cosmetic but misleading; report
  `len(all_md5s)` or the actual sig count.

---

## 5. How to select an optimal Selector

This is the central open question (CLAUDE.md research Q1), and the validation work already gives a
defensible answer and a reusable procedure.

**What the data showed (183 samples):**
- Selectors disagree strongly — mean Jaccard overlap 0.01–0.31 at k=5. The choice is *not*
  cosmetic.
- On the **unique-variance** metric (orthogonal projection, the unbiased one):
  MaxMin wins at small k (k=3: 76% vs top-var 62%); top-var edges ahead by k=5 (87% vs 84%); all
  converge to ~90% by k=10. "Top containment" — picking assemblies most similar to the query —
  loses at every k because it selects correlated, redundant references.
- The naive "sum of variances" metric inverts this ranking, which is exactly why an unbiased metric
  matters.

**Recommended selection procedure (a three-tier funnel):**

1. **Tier 0 — agreement screen (seconds).** Compute pairwise selection overlap. If all selectors
   agree (high Jaccard), pick the cheapest (random/maxmin) and stop — the choice doesn't matter for
   this dataset. The mouse data failed this screen, so proceed.

2. **Tier 1 — variance-explained proxy (seconds–minutes, no alignment).** Rank selectors by unique
   cross-sample variance explained at the adaptive k. This is the orthogonal-projection metric from
   `compare_selectors.py`. It needs only the containment (or similarity) matrix and discriminates
   selectors cleanly. Use it as the routine, scalable decision rule.

3. **Tier 2 — MAG yield per CPU-hour (hours, gold standard).** For the 2–3 finalists from Tier 1,
   run the full pipeline on one or two datasets and count CheckM2-passing MAGs (≥50% complete,
   ≤10% contaminated) per CPU-hour. This is the metric that actually matters and the one that
   *validates the proxy*. Do it once to establish that Tier-1 ranking predicts Tier-2 ranking; if
   it does, you rarely need Tier 2 again.

**The key research deliverable** is the Tier-1↔Tier-2 correlation. If unique-variance-explained
predicts CheckM2 MAG yield, you have a principled, cheap selector-choice rule and a publishable
result. If it doesn't, that itself tells you the proxy is missing something (e.g., mapping rate,
assembly fragmentation) and points to the next metric.

**Practical default to ship now:** MaxMin on containment profiles at adaptive k (≈4–5 for cohort
data). It's the robust choice across k, it's cheap, and it dominates at the small k that adaptive-k
actually selects. Make it the documented default; expose top-var as the alternative for users who
fix a larger k.

---

## 6. Additional functionality worth adding

- **`refrover rank-selectors` subcommand.** Operationalize §5 Tiers 0–1: given sketches or a
  containment matrix, run every selector at a k-range and print the agreement screen + variance
  proxy. A pre-flight that tells the user which selector to use before committing to alignment.
- **`refrover containment` subcommand** (per §4.1) — first-class containment computation.
- **Min-threshold calibration helper.** Research Q3 (what `min_jaccard`/`min_containment` to use)
  is answerable from the data: at `min_containment=0.1` some samples drop to 0 candidates, at 0.02
  the median is ~172. A small utility that plots candidate-count vs threshold and picks the knee
  would stop users from silently starving samples of references.
- **Run provenance file.** Write `params.json` per run (selector, k, thresholds, tool versions,
  input hashes). Essential for reproducibility and for the benchmark bookkeeping.
- **Resumability beyond file-existence.** A small state file (which samples are sketched/aligned/
  covered) makes the multi-hour pipeline restartable after interruption without re-globbing.
- **Smoke integration test.** Tests currently mock all external tools. One opt-in test
  (`@pytest.mark.integration`) that runs tiny real FASTA/FASTQ through bwa-mem2 + coverm would catch
  interface drift (e.g., the BAM glob in `coverage.py:34` vs. what `align` actually names files).

---

## 7. Testing & infrastructure

- **No CI.** Add a GitHub Actions workflow running `ruff`, `mypy`, and `pytest` on push. The
  `[dev]` extras already declare all three (`pyproject.toml`).
- **Selector property tests.** Every selector should satisfy invariants worth asserting once in a
  shared parametrized test: returns ≤ k IDs, returns only candidates above threshold, returns
  unique IDs, and (per §1.3) includes the query. Right now each selector has its own ad-hoc tests
  and the contract isn't enforced uniformly.
- **`coverage.py` / `align.py` have no unit tests** beyond the mocked pipeline test. The
  CoverM/BWA argument construction is exactly the kind of thing that breaks silently; a test that
  asserts the constructed command list would be cheap insurance.

---

## 8. Suggested order of work

1. Fix MetaBAT2 variance (§1.1) and the query-inclusion contract (§1.3) — correctness, small.
2. Unify the selector interface and wire `ContainmentSelector` into CLI + pipeline (§3.1, 3.2) —
   unlocks the validated method.
3. Port containment computation and the adaptive-k saturation method into the package (§4.1, 4.3).
4. Implement `benchmark.py` Tier-1 proxy + `rank-selectors` subcommand (§4.2, §6) — makes selector
   choice routine and scalable.
5. Performance passes on the two slow selectors (§2.1, 2.2).
6. CI + property tests + provenance (§7, §6).
7. Run the Tier-1↔Tier-2 validation once (§5) — the publishable result.

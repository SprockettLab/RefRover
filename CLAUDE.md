# CLAUDE.md — RefRover

RefRover is a standalone Python tool and library that computes binner-ready differential coverage tables from metagenomic assemblies and reads. It handles the full pipeline: assembly sketching (sourmash) → prototype/archetype selection → read alignment → per-contig coverage → formatted output for MetaBAT2, SemiBin2, MaxBin2, CONCOCT, or a generic TSV.

The core scientific contribution is the **adaptive prototype selection algorithm**: for each sample, RefRover intelligently selects a subset of assemblies to map against — large enough to capture differential signal, small enough to be computationally tractable, and constrained so reads actually map (minimum Jaccard similarity threshold). Multiple selection strategies are implemented so they can be benchmarked against each other across different dataset types.

RefRover was spun out of [MAGforge](https://github.com/CUMoellerLab/MAGforge), which calls it as a library between the `SOURMASH_SKETCH` and `DIFFERENTIAL_COVERAGE` pipeline stages.

---

## Scientific context

Differential-coverage-based binning (MetaBAT2, SemiBin2, etc.) requires per-contig depth estimates across multiple samples. That means reads from each sample must be mapped against a reference assembly set. The naive approach — map every sample's reads against every other sample's assembly — scales as O(n²) in alignment cost, which is prohibitive for longitudinal cohort studies (n = 20–100+ samples).

**The prototype selection problem:** For sample i, select a subset S_i of k assemblies from the full set such that:
1. **Similarity constraint**: each selected prototype has Jaccard(i, prototype) ≥ min_jaccard (reads will actually map)
2. **Diversity constraint**: prototypes are sufficiently different from each other (maximizes differential signal)
3. **k is adaptive**: chosen per-sample based on the dataset's structure, not fixed globally

The selected prototypes become the alignment reference for that sample. CoverM computes per-contig depths. The depth table is reformatted for the downstream binner.

---

## Full pipeline

```
assemblies (FASTA) + reads (FASTQ)
    ↓ [sketch]     sourmash MinHash sketches per assembly
    ↓ [select]     per-sample prototype/archetype selection
    ↓ [index]      build alignment index for selected prototypes
    ↓ [align]      BWA-MEM2 (short) or minimap2 (long/hybrid): reads → prototypes
    ↓ [coverage]   CoverM: per-contig mean depth per sample
    ↓ [format]     binner-ready output files
```

Each stage can be run independently (step-by-step mode) or the full pipeline runs end-to-end with `refrover run`.

---

## Selector strategies

Seven selection algorithms are implemented. All share the same interface (see `src/refrover/selectors/base.py`): `select(matrix, query_id) -> list[str]`, returning prototype sample_ids.

| Strategy ID | Class | Description | Mathematical basis |
|-------------|-------|-------------|-------------------|
| `random` | `RandomSelector` | Baseline: random k prototypes within Jaccard threshold | None |
| `maxmin` | `MaxMinSelector` | Greedy MaxMin: greedily maximizes minimum pairwise distance | Greedy distance maximization |
| `kmedoids` | `KMedoidsSelector` | k-medoids clustering in Jaccard space; medoids as prototypes | Minimizes within-cluster sum of distances |
| `archetype` | `ArchetypeSelector` | Archetype analysis: finds assemblies on the convex hull of the similarity space | Cutler & Breiman (1994); extremes that span the space |
| `greedy_var` | `GreedyVarSelector` | Greedy coverage-variance maximization: selects prototypes predicted to maximize cross-sample variance | Estimated via similarity proxy |
| `containment` | `ContainmentSelector` | MaxMin on cross-sample read **containment** profiles (not Jaccard); diversity in read-mapping space | Greedy distance maximization on 1 − Pearson(containment columns) |
| `feedback` | `FeedbackSelector` | Coverage-feedback: map → measure actual variance → iterate | Empirical; most accurate, most expensive (not yet implemented) |

**Containment vs. Jaccard selectors**: the Jaccard selectors operate on an assembly-vs-assembly similarity matrix; `containment` operates on a reads-vs-assembly containment matrix (`containment[i,j]` = fraction of sample i's read k-mers in assembly j), which directly measures how well reads will map. It is a `BaseSelector` subclass but is fed a different matrix — pass `--containment-matrix` (CLI) or `containment_matrix=` (`RefRoverPipeline`). On the rewilded-mouse validation it was the recommended strategy at small k. The matrix is produced by `data/mouse_rewilding/compute_containment.py` (slated to move into the package as `refrover.containment`).

**Archetype vs. k-medoids distinction**: k-medoids finds central representatives; archetypes find extreme points that together span the full diversity space. For differential coverage, span is more valuable than centrality — archetypes are the theoretically preferred approach. Benchmark both.

**Adaptive k**: The target research contribution. `k` is not fixed globally but determined per-sample from the dataset's similarity structure. The intended approach: fit an information-saturation curve (differential signal as a function of k) and select the elbow point. This is not yet implemented — the current selectors all accept an explicit `--k` parameter as a fallback.

---

## Binner output format registry

RefRover writes coverage output in the format each binner expects. Specify target binners with `--binners`; the generic TSV is always written.

| Binner | Flag value | Output file | Format notes |
|--------|-----------|-------------|-------------|
| MetaBAT2 | `metabat2` | `depth.txt` | `jgi_summarize_bam_contig_depths` format: contig, length, totalAvgDepth, sample1, sample1-var, ... |
| SemiBin2 | `semibin2` | `coverage_metabinner.tsv` | contig × sample depth matrix, no variance column |
| MaxBin2 | `maxbin2` | `{sample}.abund` per-sample abundance files | Single-column per file: contig depth |
| CONCOCT | `concoct` | `coverage_table.tsv` | contig × sample matrix, tab-separated, integer depths |
| *(always)* | — | `coverage_generic.tsv` | contig, length, sample1_depth, sample2_depth, ... |

Format implementations live in `src/refrover/formatters/`. Adding a new binner requires: (1) a new module in `formatters/`, (2) registration in `formatters/registry.py`, (3) entry in the table above.

---

## CLI design

```bash
# Full end-to-end pipeline
refrover run \
    --manifest samples.tsv \
    --selector archetype \
    --k 5 \
    --min-jaccard 0.1 \
    --aligner bwa-mem2 \
    --binners metabat2,semibin2 \
    --threads 16 \
    --outdir ./coverage/

# Step-by-step (each stage can be run independently)
refrover sketch   --manifest samples.tsv --outdir sketches/
refrover select   --sketches sketches/ --manifest samples.tsv --selector archetype --k 5
refrover align    --assignments assignments.tsv --manifest samples.tsv --outdir bams/
refrover coverage --bams bams/ --outdir raw_coverage/
refrover format   --coverage raw_coverage/ --binners metabat2,semibin2 --outdir formatted/

# Containment-based selection (computes the reads-vs-assemblies matrix, then selects)
refrover containment --read-sketches read_sigs/ --assembly-sketches sketches/ --outdir cont/
refrover select      --containment-matrix cont/containment_matrix.tsv \
                     --manifest samples.tsv --selector containment --k 5 --min-jaccard 0.05 \
                     --outdir assignments/

# Benchmark selectors against each other on a dataset with known ground truth
refrover benchmark \
    --manifest samples.tsv \
    --truth community_truth.tsv \
    --selectors random,maxmin,kmedoids,archetype,greedy_var \
    --k-range 3,5,8,10 \
    --checkm2-db /path/to/checkm2_db \
    --outdir benchmark_results/
```

### Input manifest format

Tab-separated, one row per sample. `long_reads` and `r2` are optional.

```tsv
sample_id	assembly	r1	r2	long_reads
S1_T0	assemblies/S1_T0.fasta	reads/S1_T0_R1.fastq.gz	reads/S1_T0_R2.fastq.gz
S1_T1	assemblies/S1_T1.fasta	reads/S1_T1_R1.fastq.gz	reads/S1_T1_R2.fastq.gz
S2_T0	assemblies/S2_T0.fasta				reads/S2_T0_ONT.fastq.gz
```

Rules:
- `assembly` is required for every row
- At least one of `r1` or `long_reads` required
- `r2` requires `r1`; absent `r2` = single-end short reads
- Presence of both `r1` and `long_reads` = hybrid sample (aligns both, merges depth)

### Prototype assignment output (written by `refrover select`)

```tsv
sample_id	prototype_ids	n_prototypes	mean_jaccard_to_nearest
S1_T0	S1_T0,S1_T1,S2_T0	3	0.42
S1_T1	S1_T0,S1_T1,S1_T2,S2_T0	4	0.38
```

---

## Python library interface

MAGforge (and other callers) use RefRover as a library without subprocess overhead.

```python
from refrover import RefRoverPipeline
from refrover.selectors import ArchetypeSelector

pipeline = RefRoverPipeline(
    manifest=manifest_df,
    selector=ArchetypeSelector(k=5, min_jaccard=0.1),
    binners=["metabat2", "semibin2"],
    threads=16,
    outdir=Path("coverage/"),
)
results = pipeline.run()
# results.coverage_tables: dict[str, Path]  — one per binner
# results.assignments: pd.DataFrame         — prototype assignments per sample
```

Individual stages are also importable:

```python
from refrover.sketch import sketch_assemblies
from refrover.selectors import ArchetypeSelector
from refrover.coverage import run_coverm
from refrover.formatters import format_for_binner
```

---

## Repository structure

```
refrover/
├── pyproject.toml              # build metadata, dependencies, entry points
├── CLAUDE.md                   # this file
├── PLAN.md                     # research roadmap and algorithm design notes
├── src/
│   └── refrover/
│       ├── __init__.py         # exports RefRoverPipeline, version
│       ├── cli.py              # Click entry point; subcommands: run, sketch, select, align, coverage, format, benchmark
│       ├── pipeline.py         # RefRoverPipeline orchestrator
│       ├── io.py               # manifest read/write, validation
│       ├── sketch.py           # sourmash wrapper (sketch + pairwise compare)
│       ├── similarity.py       # similarity matrix operations, Jaccard filtering, containment loader
│       ├── containment.py      # reads-vs-assemblies containment matrix (feeds ContainmentSelector)
│       ├── selectors/
│       │   ├── __init__.py     # exports all selectors + SELECTOR_REGISTRY dict
│       │   ├── base.py         # BaseSelector ABC: select(similarity_matrix, query_id) -> list[str]
│       │   ├── random.py       # RandomSelector
│       │   ├── maxmin.py       # MaxMinSelector
│       │   ├── kmedoids.py     # KMedoidsSelector (uses sklearn or sklearn-extra)
│       │   ├── archetype.py    # ArchetypeSelector (custom or py-archetypes)
│       │   ├── greedy_var.py   # GreedyVarSelector
│       │   └── feedback.py     # FeedbackSelector (calls aligner + CoverM internally)
│       ├── align.py            # BWA-MEM2 / minimap2 wrappers; returns sorted BAM paths
│       ├── coverage.py         # CoverM wrapper; returns per-contig depth DataFrame
│       ├── formatters/
│       │   ├── __init__.py     # exports format_for_binner()
│       │   ├── registry.py     # BINNER_REGISTRY: str -> FormatterClass
│       │   ├── metabat2.py
│       │   ├── semibin2.py
│       │   ├── maxbin2.py
│       │   ├── concoct.py
│       │   └── generic.py
│       └── benchmark.py        # evaluation: MAG yield per compute vs. ground truth
├── tests/
│   ├── conftest.py             # fixtures: tiny FASTA + FASTQ pairs, mock sketches
│   ├── test_selectors/
│   │   ├── test_random.py
│   │   ├── test_maxmin.py
│   │   ├── test_kmedoids.py
│   │   ├── test_archetype.py
│   │   └── test_greedy_var.py
│   ├── test_formatters/
│   │   ├── test_metabat2.py
│   │   └── test_generic.py
│   └── test_pipeline.py        # end-to-end with mock external tools
└── benchmarks/
    ├── simulate_community.py   # generate synthetic metagenome with known composition
    └── run_benchmark.py        # run all selectors, collect MAG yield + runtime
```

---

## Dependencies

**Required Python packages** (declare in `pyproject.toml`):
- `sourmash` — sketching and similarity computation
- `click` — CLI
- `pandas` — manifest and coverage table I/O
- `numpy` — similarity matrix operations
- `scipy` — distance metrics, optimization (used by kmedoids, archetype)
- `scikit-learn` or `scikit-learn-extra` — k-medoids (sklearn-extra has `KMedoids`)

**Required external tools** (must be on PATH or specified via `--bwa-path` / `--minimap2-path` etc.):
- `bwa-mem2` (short/hybrid read alignment; fallback to `bwa`)
- `minimap2` (long read alignment)
- `samtools` (BAM sort/index)
- `coverm` (per-contig coverage computation)

**Optional:**
- `checkm2` — required only for `refrover benchmark`
- `py-archetypes` — pure-Python archetype analysis; if absent, RefRover uses its own implementation in `selectors/archetype.py`

---

## Development environment

```bash
conda create -n refrover-dev python=3.11 sourmash bwa-mem2 minimap2 samtools coverm
conda activate refrover-dev
pip install -e ".[dev]"   # installs refrover + pytest, ruff, mypy

refrover --help            # verify CLI entry point works
pytest tests/              # unit tests (all should pass without external tools via mocking)
```

---

## Key design decisions

**Selector interface contract**: `BaseSelector.select(sim_matrix: pd.DataFrame, query_id: str) -> list[str]`. Takes the full pairwise similarity matrix and the ID of the query sample; returns an ordered list of prototype assembly IDs. This keeps selectors stateless and testable without external tools.

**Adaptive k (not yet implemented)**: The current selectors all require an explicit `--k`. The intended design: for each query, compute the marginal gain in predicted differential signal as k increases, and stop at the elbow. This requires a signal prediction model — deferred to after the fixed-k selectors are benchmarked.

**Coverage merging for hybrid samples**: For samples with both short and long reads, align separately, then take the mean depth per contig weighted by read count. This is handled in `coverage.py`.

**Idempotent stages**: Each stage writes its output to a predictable path under `--outdir`. Re-running a stage skips existing outputs (file-existence check). The `--force` flag disables this.

**No Nextflow dependency**: RefRover is pure Python + subprocess calls. It has no Nextflow or nf-core dependency and can run in any environment where the external tools are available.

---

## Relationship to MAGforge

In MAGforge, RefRover is called between `SOURMASH_SKETCH` (Phase 8) and `DIFFERENTIAL_COVERAGE` (Phase 10):

```nextflow
// MAGforge workflow snippet (conceptual)
SOURMASH_SKETCH(ch_assemblies)
ch_assignments = REFROVER_SELECT(
    SOURMASH_SKETCH.out.sketches,
    ch_reads_manifest,
    params.mapping_mode,   // 'adaptive' | 'all' | 'group'
    params.refrover_selector
)
DIFFERENTIAL_COVERAGE(ch_assignments, ch_reads)
```

When `params.mapping_mode = 'group'`, RefRover is skipped and Hub 1 BAMs are reused directly. When `mapping_mode = 'adaptive'` (default), RefRover runs with the configured selector.

MAGforge declares RefRover as a Python dependency in its environment files. It calls RefRover via the Python library interface, not subprocess.

---

## Research questions (open)

1. **Which selector wins?** Archetype analysis is theoretically preferred; empirical benchmarking across real and simulated datasets required. Primary metric: CheckM2-passing MAGs (≥50% completeness, ≤10% contamination) per CPU-hour.

2. **Adaptive k algorithm**: What signal-saturation model best predicts the elbow? Options: information gain, coverage-variance curve fitting, silhouette-based stopping criterion.

3. **Minimum Jaccard threshold**: What value ensures reads actually map at useful rates? Likely dataset-dependent (species richness, read length, assembly quality). Needs characterization.

4. **Feedback selector viability**: The feedback approach (map → measure → iterate) is the most accurate but involves multiple mapping rounds. Is the compute cost justified? Under what dataset conditions?

5. **Long-read and hybrid behavior**: The sketching and similarity metrics are validated for short-read assemblies. Characterize whether they behave differently for Flye long-read or metaSPAdes hybrid assemblies.

#!/usr/bin/env bash
# MAG-quality benchmark — one SLURM task per focal assembly (or per shard).
#
# Preferred usage — one focal per task (maximally parallel):
#   python -c "
#     import pandas as pd, random; random.seed(42)
#     ids = pd.read_csv('samples.tsv', sep='\t')['sample_id'].tolist()
#     print('\n'.join(random.sample(ids, 30)))   # or len(ids) for all
#   " > work/focals_30.txt
#
#   N=$(wc -l < work/focals_30.txt)
#   FOCALS_FILE=work/focals_30.txt sbatch --array=1-${N}%50 scripts/slurm/04_benchmark.sh
#
# When FOCALS_FILE is set, each array task handles exactly one focal (the line
# at position SLURM_ARRAY_TASK_ID). When it is absent, the old --shard i/N
# round-robin over all manifest samples is used instead.
#
# Depends on 03_build_matrices.sh completing first.
#
# Environment variables (override any with SBATCH --export or export before sbatch):
#   MANIFEST        path to samples.tsv
#   ASM_DIR         directory of per-sample assembly FASTAs (stem = sample_id)
#   MATRIX_DIR      directory containing jaccard_matrix.tsv, containment_matrix.tsv
#   GTDB_DIR        directory containing gtdb_matrix.tsv (from 03c_gtdb_matrix.sh);
#                   optional — enables taxonomy_stratified when present
#   FOCALS_FILE     file with one focal sample_id per line (enables 1-focal-per-task mode)
#   BENCHMARK_DIR   output directory for benchmark cells
#   CHECKM2_DB      path to CheckM2 DIAMOND database
#   RULES           comma-separated rule names (default: random,maxmin,css)
#   K_RANGE         comma-separated k values; 'adaptive' accepted (default: 5,10,15,20,adaptive)
#   THREADS         threads per task (default: 32; set CPUs to match in sbatch)
#
#SBATCH --job-name=rr_benchmark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
ASM_DIR="${ASM_DIR:-assemblies}"
MATRIX_DIR="${MATRIX_DIR:-matrices}"
GTDB_DIR="${GTDB_DIR:-}"
FOCALS_FILE="${FOCALS_FILE:-}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
CHECKM2_DB="${CHECKM2_DB:-}"
RULES="${RULES:-random,maxmin,css}"
K_RANGE="${K_RANGE:-5,10,15,20,adaptive}"
THREADS="${THREADS:-32}"

N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

# Locate the checkm2 binary (lives in its own conda env to avoid dep conflicts).
CHECKM2_BIN=$(conda run -n checkm2 which checkm2 2>/dev/null || echo "")
if [[ -z "$CHECKM2_BIN" ]]; then
    echo "ERROR: checkm2 not found in conda env 'checkm2'. Install it first:" >&2
    echo "  conda env create -f environment-checkm2.yml" >&2
    echo "  conda activate checkm2 && checkm2 database --download --path ~/checkm2_db" >&2
    exit 1
fi

# Build optional flags.
DB_FLAG=""
if [[ -n "$CHECKM2_DB" ]]; then
    DB_FLAG="--checkm2-db $CHECKM2_DB"
fi

# Build optional flags.
CONT_FLAG=""
if [[ -f "$MATRIX_DIR/containment_matrix.tsv" ]]; then
    CONT_FLAG="--containment-matrix $MATRIX_DIR/containment_matrix.tsv"
fi

GTDB_FLAG=""
if [[ -n "$GTDB_DIR" && -f "$GTDB_DIR/gtdb_matrix.tsv" ]]; then
    GTDB_FLAG="--gtdb-matrix $GTDB_DIR/gtdb_matrix.tsv"
fi

# One-focal-per-task mode vs legacy shard mode.
if [[ -n "$FOCALS_FILE" ]]; then
    FOCAL_ID=$(sed -n "${I}p" "$FOCALS_FILE")
    if [[ -z "$FOCAL_ID" ]]; then
        echo "No focal at line ${I} of ${FOCALS_FILE}; nothing to do." && exit 0
    fi
    # Write a single-line focals file for this task so --focals accepts it.
    TMP_FOCALS=$(mktemp)
    echo "$FOCAL_ID" > "$TMP_FOCALS"
    SHARD_FLAG=""
    FOCALS_FLAG="--focals $TMP_FOCALS"
else
    TMP_FOCALS=""
    SHARD_FLAG="--shard ${I}/${N}"
    FOCALS_FLAG=""
fi

conda run -n refrover-benchmark \
    refrover benchmark \
        --manifest          "$MANIFEST" \
        --assemblies-dir    "$ASM_DIR" \
        --jaccard-matrix    "$MATRIX_DIR/jaccard_matrix.tsv" \
        --rules             "$RULES" \
        --k-range           "$K_RANGE" \
        --threads           "$THREADS" \
        --outdir            "$BENCHMARK_DIR" \
        --checkm2-path      "$CHECKM2_BIN" \
        $CONT_FLAG \
        $GTDB_FLAG \
        $DB_FLAG \
        $FOCALS_FLAG \
        $SHARD_FLAG

[[ -n "$TMP_FOCALS" ]] && rm -f "$TMP_FOCALS"

if [[ -n "$FOCALS_FILE" ]]; then
    echo "Focal ${FOCAL_ID} (task ${I}) complete."
else
    echo "Shard ${I}/${N} complete."
fi

#!/usr/bin/env bash
# MAG-quality benchmark — one SLURM task per shard of focal assemblies.
#
# Each task runs a round-robin slice of focal assemblies across all
# (matrix × rule × k) combinations. When all tasks finish, run
# 'refrover aggregate-results' to merge the per-cell result.json files.
#
# Usage:
#   N_FOCALS=$(wc -l < focals.txt)   # or use N_SAMPLES from manifest
#   sbatch --array=1-${N_FOCALS}%50 scripts/slurm/04_benchmark.sh
#
# The %50 cap limits concurrent tasks to 50 (adjust for your cluster policy).
# Depends on 03_build_matrices.sh completing first.
#
# Environment variables (override any with SBATCH --export or export before sbatch):
#   MANIFEST        path to samples.tsv
#   ASM_DIR         directory of per-sample assembly FASTAs (stem = sample_id)
#   MATRIX_DIR      directory containing jaccard_matrix.tsv, containment_matrix.tsv
#   BENCHMARK_DIR   output directory for benchmark cells
#   CHECKM2_DB      path to CheckM2 DIAMOND database
#   RULES           comma-separated rule names (default: random,maxmin,css)
#   K_RANGE         comma-separated k values   (default: 3,5,8,10)
#   THREADS         threads per task            (default: 8)
#
#SBATCH --job-name=rr_benchmark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
ASM_DIR="${ASM_DIR:-assemblies}"
MATRIX_DIR="${MATRIX_DIR:-matrices}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
CHECKM2_DB="${CHECKM2_DB:-}"
RULES="${RULES:-random,maxmin,css}"
K_RANGE="${K_RANGE:-3,5,8,10}"
THREADS="${THREADS:-8}"

N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

# Build the --checkm2-db flag only when a db path is set.
DB_FLAG=""
if [[ -n "$CHECKM2_DB" ]]; then
    DB_FLAG="--checkm2-db $CHECKM2_DB"
fi

# CheckM2 runs inside its own conda env; pass its binary path so the
# refrover-benchmark env can invoke it via subprocess.
CHECKM2_BIN=$(conda run -n checkm2 which checkm2 2>/dev/null || echo "checkm2")

conda run -n refrover-benchmark \
    refrover benchmark \
        --manifest          "$MANIFEST" \
        --assemblies-dir    "$ASM_DIR" \
        --jaccard-matrix    "$MATRIX_DIR/jaccard_matrix.tsv" \
        --containment-matrix "$MATRIX_DIR/containment_matrix.tsv" \
        --rules             "$RULES" \
        --k-range           "$K_RANGE" \
        --threads           "$THREADS" \
        --shard             "${I}/${N}" \
        --outdir            "$BENCHMARK_DIR" \
        $DB_FLAG

echo "Shard ${I}/${N} complete."

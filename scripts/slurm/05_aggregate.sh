#!/usr/bin/env bash
# Aggregate all per-cell result.json files into one benchmark_results.tsv.
#
# Run this once all 04_benchmark.sh array tasks have finished:
#   sbatch --dependency=afterok:<benchmark_job_id> scripts/slurm/05_aggregate.sh
#
# Or run manually:
#   bash scripts/slurm/05_aggregate.sh
#
#SBATCH --job-name=rr_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail
mkdir -p logs

BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"

conda run -n refrover-benchmark \
    refrover aggregate-results \
        --outdir "$BENCHMARK_DIR"

echo "Done. Results: ${BENCHMARK_DIR}/benchmark_results.tsv"

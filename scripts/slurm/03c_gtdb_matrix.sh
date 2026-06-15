#!/usr/bin/env bash
# Aggregate per-sample GTDB gather CSVs into the abundance matrix.
#
# Run this once after all 03b_gtdb_gather.sh array tasks have finished.
# Produces:
#   gtdb/gtdb_matrix.tsv   (samples × taxa abundance; input for 04_benchmark.sh)
#   gtdb/clades.tsv        (dominant genus per sample; for inspection)
#
# Usage:
#   sbatch scripts/slurm/03c_gtdb_matrix.sh
#
# Environment variables:
#   MANIFEST     path to samples.tsv
#   GTDB_DIR     must match GTDB_DIR from 03b_gtdb_gather.sh (default: gtdb)
#
#SBATCH --job-name=rr_gtdb_matrix
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gtdb_matrix_%j.out
#SBATCH --error=logs/gtdb_matrix_%j.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
GTDB_DIR="${GTDB_DIR:-gtdb}"

conda run -n refrover-benchmark \
    refrover gtdb-matrix \
        --gather-dir "$GTDB_DIR/gather" \
        --manifest   "$MANIFEST" \
        --outdir     "$GTDB_DIR"

echo "GTDB matrix complete → ${GTDB_DIR}/gtdb_matrix.tsv"
echo "Pass to benchmark with: --gtdb-matrix ${GTDB_DIR}/gtdb_matrix.tsv"

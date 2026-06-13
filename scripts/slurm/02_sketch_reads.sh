#!/usr/bin/env bash
# Sketch reads per sample — one SLURM task per sample.
#
# Must run after 01_sketch_assemblies.sh completes (or independently; these two
# jobs don't depend on each other, so they can run in parallel).
#
# Usage:
#   sbatch --array=1-<N_SAMPLES> scripts/slurm/02_sketch_reads.sh \
#       --manifest samples.tsv --outdir sketches/reads
#
#SBATCH --job-name=rr_sketch_reads
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sketch_reads_%A_%a.out
#SBATCH --error=logs/sketch_reads_%A_%a.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
OUTDIR="${OUTDIR:-sketches/reads}"
KSIZE="${KSIZE:-31}"
SCALED="${SCALED:-1000}"

N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

conda run -n refrover-benchmark \
    refrover sketch-reads \
        --manifest  "$MANIFEST" \
        --outdir    "$OUTDIR" \
        --ksize     "$KSIZE" \
        --scaled    "$SCALED" \
        --shard     "${I}/${N}"

#!/usr/bin/env bash
# Sketch assembly FASTAs — one SLURM task per sample.
#
# Usage:
#   sbatch --array=1-<N_SAMPLES> scripts/slurm/01_sketch_assemblies.sh \
#       --manifest samples.tsv --outdir sketches/assemblies
#
# Each task processes 1/N of the samples (round-robin) so tasks never conflict.
# Re-submit with the same command to pick up any stragglers (idempotent).
#
# Tune these defaults for your cluster:
#SBATCH --job-name=rr_sketch_asm
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sketch_asm_%A_%a.out
#SBATCH --error=logs/sketch_asm_%A_%a.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
OUTDIR="${OUTDIR:-sketches/assemblies}"
KSIZE="${KSIZE:-31}"
SCALED="${SCALED:-1000}"

N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

conda run -n refrover-benchmark \
    refrover sketch \
        --manifest  "$MANIFEST" \
        --outdir    "$OUTDIR" \
        --ksize     "$KSIZE" \
        --scaled    "$SCALED" \
        --shard     "${I}/${N}"

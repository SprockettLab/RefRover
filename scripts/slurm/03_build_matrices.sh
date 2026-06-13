#!/usr/bin/env bash
# Compute the Jaccard and containment matrices — single jobs (fast).
#
# Depends on 01 and 02 finishing first. Submit after both sketch jobs complete:
#   sbatch --dependency=afterok:<job1_id>,<job2_id> scripts/slurm/03_build_matrices.sh
#
# Or just run manually once the sketch directories are populated:
#   bash scripts/slurm/03_build_matrices.sh
#
#SBATCH --job-name=rr_matrices
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/matrices_%j.out
#SBATCH --error=logs/matrices_%j.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
ASM_SIGS="${ASM_SIGS:-sketches/assemblies}"
READ_SIGS="${READ_SIGS:-sketches/reads}"
MATRIX_DIR="${MATRIX_DIR:-matrices}"
KSIZE="${KSIZE:-31}"

conda run -n refrover-benchmark \
    refrover jaccard \
        --assembly-sketches "$ASM_SIGS" \
        --ksize             "$KSIZE" \
        --outdir            "$MATRIX_DIR"

conda run -n refrover-benchmark \
    refrover containment \
        --read-sketches     "$READ_SIGS" \
        --assembly-sketches "$ASM_SIGS" \
        --ksize             "$KSIZE" \
        --outdir            "$MATRIX_DIR"

echo "Matrices written to $MATRIX_DIR"

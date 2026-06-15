#!/usr/bin/env bash
# GTDB taxonomic gather — one SLURM task per shard of samples.
#
# Runs 'refrover gtdb-gather' for a round-robin slice of samples (one slice
# per SLURM array task). When all tasks finish, run 03c_gtdb_matrix.sh to
# aggregate the per-sample CSVs into the matrix used by 04_benchmark.sh.
#
# Usage:
#   N=$(wc -l < samples.tsv)   # one task per sample; adjust %N cap for policy
#   sbatch --array=1-${N}%20 scripts/slurm/03b_gtdb_gather.sh
#
# Typical runtime: 2–10 min per sample (depends on GTDB DB size and read depth).
# Memory: ~8 GB per task with the representative-genome DB (~3 GB on disk).
#
# Download the GTDB sourmash database first:
#   curl -L https://osf.io/th2my/download -o gtdb-rs214-reps.k31.zip
#
# Environment variables (override with SBATCH --export or export before sbatch):
#   MANIFEST        path to samples.tsv
#   READ_SIGS_DIR   directory of per-sample read .sig files (from 02_sketch_reads.sh)
#   GTDB_DB         path to GTDB sourmash .zip database
#   GTDB_DIR        output directory for gather CSVs and the final matrix
#   KSIZE           k-mer size (default 31; must match read sketches and GTDB DB)
#
#SBATCH --job-name=rr_gtdb
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gtdb_%A_%a.out
#SBATCH --error=logs/gtdb_%A_%a.err

set -euo pipefail
mkdir -p logs

MANIFEST="${MANIFEST:-samples.tsv}"
READ_SIGS_DIR="${READ_SIGS_DIR:-read_sigs}"
GTDB_DB="${GTDB_DB:-}"
GTDB_DIR="${GTDB_DIR:-gtdb}"
KSIZE="${KSIZE:-31}"

if [[ -z "$GTDB_DB" ]]; then
    echo "ERROR: set GTDB_DB to the path of the GTDB sourmash .zip database." >&2
    echo "  Download: curl -L https://osf.io/th2my/download -o gtdb-rs214-reps.k31.zip" >&2
    exit 1
fi
if [[ ! -f "$GTDB_DB" ]]; then
    echo "ERROR: GTDB_DB not found: $GTDB_DB" >&2
    exit 1
fi

N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

GATHER_DIR="${GTDB_DIR}/gather"

conda run -n refrover-benchmark \
    refrover gtdb-gather \
        --manifest      "$MANIFEST" \
        --read-sketches "$READ_SIGS_DIR" \
        --gtdb-db       "$GTDB_DB" \
        --outdir        "$GATHER_DIR" \
        --ksize         "$KSIZE" \
        --shard         "${I}/${N}"

echo "GTDB gather shard ${I}/${N} complete → ${GATHER_DIR}"

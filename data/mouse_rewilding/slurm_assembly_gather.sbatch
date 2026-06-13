#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm array job: assembly-side GTDB gather against the FULL database, in
# parallel across the cluster (PLAN.md §10 Q6 — the sub-DB-vs-full-DB validation).
#
# The full 143k-sig GTDB scan is ~26 min/assembly → ~80 h serial for 182
# assemblies. This array runs N shards concurrently; each gathers a disjoint
# round-robin slice (run_assembly_gather.py --shard i/N) and writes only its own
# per-assembly CSVs. Aggregation into gtdb_assembly_matrix.tsv is a SEPARATE,
# single, post-step (see the dependency job at the bottom) so no task ever writes
# a partial matrix.
#
# Each shard is resume-capable: re-running skips assemblies whose CSV already
# exists, so a requeued/preempted task picks up where it left off.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   N=20                                   # number of shards == array width
#   GA=$(sbatch --array=1-$N --export=ALL,N=$N slurm_assembly_gather.sbatch | awk '{print $4}')
#   # aggregate once, only after ALL shards succeed:
#   sbatch --dependency=afterok:$GA --export=ALL,AGGREGATE=1 slurm_assembly_gather.sbatch
#
#   Keep --array=1-N and the exported N in sync (Slurm can't read a var in the
#   #SBATCH --array directive, so pass --array on the command line as above).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=refrover-asm-gather
#SBATCH --output=logs/asm_gather_%A_%a.out
#SBATCH --error=logs/asm_gather_%A_%a.err
#SBATCH --time=12:00:00            # generous: ~9 assemblies/shard × ~26 min ≈ 4 h
#SBATCH --cpus-per-task=4          # sourmash gather is light; the DB scan is the cost
#SBATCH --mem=16G                  # full GTDB zip is ~3.9 GB; headroom for the index
#SBATCH --partition=normal         # <-- set to your cluster's partition

set -euo pipefail
mkdir -p logs

# ── Environment (edit for your cluster) ──────────────────────────────────────
# module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate refrover-dev       # must have sourmash + pandas on PATH

REPO=/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover
cd "$REPO/data/mouse_rewilding"

if [[ "${AGGREGATE:-0}" == "1" ]]; then
    echo "Aggregation pass (all shards finished) ..."
    python run_assembly_gather.py --full-db --aggregate-only
else
    : "${N:?set N to the array width, e.g. --export=ALL,N=20}"
    echo "Shard ${SLURM_ARRAY_TASK_ID}/${N} on $(hostname)"
    python run_assembly_gather.py --full-db --shard "${SLURM_ARRAY_TASK_ID}/${N}"
fi

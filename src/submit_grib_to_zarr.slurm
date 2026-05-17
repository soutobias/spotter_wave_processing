#!/bin/bash
#SBATCH --job-name=merge_waves
#SBATCH --time=04:00:00
#SBATCH --account=paidiver
#SBATCH --qos=high
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/grib_to_zarr_%j.out
#SBATCH --error=logs/grib_to_zarr_%j.err

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export ECCODES_THREADS="${SLURM_CPUS_PER_TASK}"

PYTHON=/home/users/tobfer/micromamba/envs/waves/bin/python

INPUT_PATH="/gws/ssde/j25b/atlantis_vis/waves"
OUTPUT_ZARR="s3://waves/era5_parametric_zarr.zarr"
CREDENTIALS_JSON="/home/users/tobfer/credentials_paidiver.json"
LOCAL_OUTPUT_DIR="/gws/ssde/j25b/atlantis_vis/waves/output_zarr_parametric"

echo "Python executable:"
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" --version

echo "Checking required packages"
"$PYTHON" -c "import xarray, dask, zarr, s3fs; print('packages ok')"

mkdir -p "${LOCAL_OUTPUT_DIR}"

"$PYTHON" grib_to_local_zarr.py \
  --input-dir "${INPUT_PATH}" \
  --local-output-dir "${LOCAL_OUTPUT_DIR}" \
  --start-year 2021 \
  --end-year 2026

echo "Done"

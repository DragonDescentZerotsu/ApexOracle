#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 96:00:00                    # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=eval_lm1b_ppl_${MODEL} \
  eval_lm1b_ppl.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (choices: ar, mdlm, udlm)
#  - SEED (optional: default = 1)

if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

if [ "${MODEL}"  = "ar" ]; then
  PARAMETERIZATION="ar"
  DIFFUSION="absorbing_state"
  TRAIN_T=0
  ZERO_RECON_LOSS=False
  TIME_COND=False
  BATCH_SIZE=128
  CKPT="${PWD}/outputs/lm1b/ar"
elif [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION="subs"
  DIFFUSION="absorbing_state"
  TRAIN_T=0
  ZERO_RECON_LOSS=False
  TIME_COND=False
  BATCH_SIZE=128
  CKPT="${PWD}/outputs/lm1b/mdlm"
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION="d3pm"
  DIFFUSION="uniform"
  TRAIN_T=0
  ZERO_RECON_LOSS=True
  TIME_COND=True
  BATCH_SIZE=64
  CKPT="${PWD}/outputs/lm1b/udlm"
else
  echo "Invalid MODEL: ${MODEL}"
  exit 1
fi

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode="ppl_eval" \
    eval.checkpoint_path="${CKPT}/checkpoints/last.ckpt" \
    eval.generate_samples=False \
    loader.eval_batch_size=${BATCH_SIZE} \
    data=lm1b \
    data.wrap=False \
    backbone=dit \
    model=small \
    model.length=128 \
    training.guidance=null \
    parameterization=${PARAMETERIZATION} \
    diffusion=${DIFFUSION} \
    time_conditioning=${TIME_COND} \
    zero_recon_loss=${ZERO_RECON_LOSS} \
    T=${TRAIN_T}

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
  --job-name=eval_text8_gen_ppl_${MODEL} \
  eval_text8_gen_ppl.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (choices: ar, mdlm, udlm)
#  - SAMPLING_STEPS (optional: default = 128)
#  - SEED (optional: default = 1)

if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${SAMPLING_STEPS}" ]; then
  SAMPLING_STEPS=128
fi
if [ -z "${SEED}" ]; then
  SEED=1
fi

if [ "${MODEL}"  = "ar" ]; then
  parameterization="ar"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=False
  CKPT="${PWD}/outputs/text8/ar"
elif [ "${MODEL}" = "mdlm" ]; then
  parameterization="subs"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=True
  CKPT="${PWD}/outputs/text8/mdlm"
elif [ "${MODEL}" = "udlm" ]; then
  parameterization="d3pm"
  diffusion="uniform"
  TRAIN_T=0
  time_conditioning=True
  sampling_use_cache=False
  CKPT="${PWD}/outputs/text8/udlm"
else
  echo "Invalid MODEL: ${MODEL}"
  exit 1
fi
generated_seqs_path="${CKPT}/samples-text8-gen-ppl-eval-_T-${SAMPLING_STEPS}_seed-${SEED}.json"

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    hydra.run.dir="${CKPT}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="${CKPT}/checkpoints/best.ckpt" \
    data=text8 \
    backbone=dit \
    model=small \
    model.length=256 \
    training.guidance=null \
    parameterization=${parameterization} \
    diffusion=${diffusion} \
    time_conditioning=${time_conditioning} \
    T=${TRAIN_T} \
    sampling.num_sample_batches=32 \
    sampling.batch_size=32 \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.use_cache=${sampling_use_cache} \
    eval.generated_samples_path=${generated_seqs_path} \
    +eval.generative_ppl_model_name_or_path="gpt2-large"

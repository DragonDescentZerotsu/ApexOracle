#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# NOTE: Need to set the (local) dataset path for downloaded cifar-10 data
DATASET_PATH="<path_to_datasets>"

<<comment
#  Usage:
cd scripts/
MODEL=<mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_cifar10_${MODEL} \
  train_cifar10_unet_guidance.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

RUN_NAME="${MODEL}"
T=0
if [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION=subs
  DIFFUSION="absorbing_state"
  ZERO_RECON_LOSS=False
  time_conditioning=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION=d3pm
  DIFFUSION="uniform"
  ZERO_RECON_LOSS=True
  time_conditioning=True
  sampling_use_cache=False
else
  echo "MODEL must be one of mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  is_vision=True \
  diffusion=${DIFFUSION} \
  parameterization=${PARAMETERIZATION} \
  T=${T} \
  time_conditioning=${time_conditioning} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=cifar10 \
  data.train=${DATASET_PATH} \
  data.valid=${DATASET_PATH} \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=64 \
  backbone=unet \
  model=unet \
  optim.lr=2e-4 \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  trainer.max_steps=300_000 \
  trainer.val_check_interval=10_000 \
  +trainer.check_val_every_n_epoch=null \
  training.guidance.cond_dropout=0.1 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="cifar10_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/cifar10/${RUN_NAME}"

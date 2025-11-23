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

<<comment
#  Usage:
cd scripts/
DIFFUSION=<absorbing_state|uniform>
sbatch \
  --export=ALL,DIFFUSION=${DIFFUSION} \
  --job-name=train_ten_species_classifier_${DIFFUSION} \
  train_ten_species_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - DIFFUSION (absorbing_state or uniform)
if [ -z "${DIFFUSION}" ]; then
  echo "DIFFUSION is not set"
  exit 1
fi
T=0
RUN_NAME="${DIFFUSION}_T-${T}"

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  mode=train_classifier \
  diffusion=${DIFFUSION} \
  T=${T} \
  data=ten_species \
  loader.global_batch_size=32 \
  loader.eval_global_batch_size=64 \
  classifier_backbone=dimamba \
  classifier_model=tiny-dimamba-classifier \
  classifier_model.bidirectional=True \
  classifier_model.bidirectional_strategy=add \
  classifier_model.bidirectional_weight_tie=True \
  model=dimamba \
  backbone=dimamba \
  model.length=32768 \
  model.bidirectional=True \
  model.bidirectional_strategy=add \
  model.bidirectional_weight_tie=True \
  optim.lr=2e-3 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=3000 \
  lr_scheduler.lr_min=2e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=6_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=3_000 \
  trainer.max_steps=30_000 \
  wandb.group=train_classifier \
  wandb.name="ten_species-classifier_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/ten_species/classifier/${RUN_NAME}"

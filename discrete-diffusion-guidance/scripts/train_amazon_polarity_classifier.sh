#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
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
  --job-name=train_amazon_classifier_${DIFFUSION} \
  train_amazon_polarity_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - DIFFUSION (absorbing_state or uniform)
#  - PROP (qed or ring_count)
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
  data=amazon_polarity \
  data.wrap=False \
  data.tokenizer_name_or_path=bert-base-uncased \
  data.label_col=label \
  data.num_classes=2 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=1024 \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=128 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1_000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=40_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.max_steps=400_000 \
  trainer.val_check_interval=1.0 \
  wandb.group=train_classifier \
  wandb.name="amazon_polarity-classifier_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/amazon_polarity/classifier/${RUN_NAME}"

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
PROP=<qed|ring_count>
sbatch \
  --export=ALL,PROP=${PROP} \
  --job-name=train_qm9_fudge_classifier_${PROP} \
  train_qm9_fudge_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL

# Expecting:
#  - PROP (qed or ring_count)
if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
LABEL_SMOOTHING=FALSE
RUN_NAME="${PROP}"

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  mode=train_classifier \
  +is_fudge_classifier=True \
  +use_label_smoothing=${LABEL_SMOOTHING} \
  parameterization=ar \
  data=qm9 \
  data.label_col="${PROP}" \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  classifier_model=tiny-classifier \
  classifier_backbone=dit \
  classifier_model.pooling=no_pooling \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance=null \
  +training.use_label_smoothing=${LABEL_SMOOTHING} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25_000 \
  wandb.group=train_classifier \
  wandb.name="qm9-fudge_classifier_${RUN_NAME}" \
  hydra.run.dir="./outputs/qm9/fudge_classifier/${RUN_NAME}"

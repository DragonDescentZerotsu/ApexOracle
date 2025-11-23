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
MODEL=<ar|mdlm|udlm>
PROP=<qed|ring_count>
sbatch \
  --export=ALL,MODEL=${MODEL},PROP=${PROP} \
  --job-name=train_qm9_pplm_classifier_${PROP}_${MODEL} \
  train_qm9_pplm_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL

# Expecting:
#  - MODEL (ar, mdlm, or udlm)
#  - PROP (qed or ring_count)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
LABEL_SMOOTHING=FALSE
RUN_NAME="${PROP}_${MODEL}"

if [ "${MODEL}" = "ar" ]; then
  # AR
  PARAMETERIZATION="ar"
  PRETRAINED_PATH="${PWD}/outputs/qm9/${MODEL}_no-guidance/checkpoints/best.ckpt"
  POOLING="attention_mean"
  # dummy properties
  DIFFUSION="absorbing_state"
  T=0
  TIME_COND=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  PRETRAINED_PATH="${PWD}/outputs/qm9/${MODEL}_no-guidance/checkpoints/best.ckpt"
  POOLING="mean"
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  PRETRAINED_PATH="${PWD}/outputs/qm9/${MODEL}_no-guidance/checkpoints/best.ckpt"
  POOLING="mean"
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  mode=train_classifier \
  +is_pplm_classifier=True \
  +use_label_smoothing=${LABEL_SMOOTHING} \
  eval.checkpoint_path="${PRETRAINED_PATH}" \
  parameterization=${PARAMETERIZATION} \
  time_conditioning=${TIME_COND} \
  diffusion=${DIFFUSION} \
  T=${T} \
  data=qm9 \
  data.label_col="${PROP}" \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  classifier_model=small-classifier \
  classifier_backbone=dit \
  classifier_model.pooling=${POOLING} \
  model.length=32 \
  +classifier_model.freeze_encoder=True \
  +classifier_model.use_encoder_ema=True \
  optim.lr=3e-5 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-7 \
  training.guidance=null \
  +training.use_label_smoothing=${LABEL_SMOOTHING} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25_000 \
  wandb.group=train_classifier \
  wandb.name="qm9-pplm_classifier_${RUN_NAME}" \
  hydra.run.dir="./outputs/qm9/pplm_classifier/${RUN_NAME}"

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
  --job-name=train_qm9_${PROP}_guidance_${MODEL} \
  train_qm9_guidance.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (ar, mdlm, udlm)
#  - PROP (qed or ring_count)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
RUN_NAME="${MODEL}_${PROP}"

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
  sampling_use_cache=False
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=qm9 \
  data.label_col=${PROP} \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  eval.generate_samples=True \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  backbone="dit" \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance.cond_dropout=0.1 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  training.compute_loss_on_pad_tokens=True \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  sampling.num_sample_batches=1 \
  sampling.batch_size=1 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=32 \
  wandb.name="qm9_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/qm9/${RUN_NAME}"

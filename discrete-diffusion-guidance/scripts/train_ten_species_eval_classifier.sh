#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
sbatch \
  --export=ALL \
  --job-name=train_ten_species_eval_classifier \
  train_ten_species_eval_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1

RUN_NAME="hyenadna-small-32k_from-scratch_nlayer-8"

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  +is_eval_classifier=True \
  mode=train_classifier \
  trainer.max_steps=30_000 \
  trainer.val_check_interval=1_000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  optim.lr=6e-5 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=3000 \
  lr_scheduler.lr_min=6e-7 \
  loader.global_batch_size=32 \
  loader.eval_global_batch_size=64 \
  data=ten_species \
  classifier_model=hyenadna-classifier \
  classifier_model.hyena_model_name_or_path="LongSafari/hyenadna-small-32k-seqlen-hf" \
  classifier_model.n_layer=8 \
  classifier_backbone=hyenadna \
  model.length=32768 \
  diffusion=null \
  T=null \
  wandb.name="ten_species_eval-classifier_${RUN_NAME}" \
  wandb.group=train_classifier \
  hydra.run.dir="./outputs/ten_species/eval_classifier/${RUN_NAME}"
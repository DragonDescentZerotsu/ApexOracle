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
sbatch \
  --export=ALL \
  --job-name=train_ten_species_fudge_classifier_${PROP} \
  train_ten_species_fudge_classifier.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL

LABEL_SMOOTHING=FALSE

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  mode=train_classifier \
  +is_fudge_classifier=True \
  +use_label_smoothing=${LABEL_SMOOTHING} \
  parameterization=ar \
  data=ten_species \
  loader.global_batch_size=32 \
  loader.eval_global_batch_size=64 \
  classifier_model=tiny-dimamba-classifier \
  classifier_backbone=dimamba \
  classifier_model.bidirectional=False \
  classifier_model.bidirectional_strategy=null \
  classifier_model.bidirectional_weight_tie=null \
  classifier_model.pooling=no_pooling \
  model.length=32768 \
  optim.lr=2e-3 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=3000 \
  lr_scheduler.lr_min=2e-6 \
  training.guidance=null \
  +training.use_label_smoothing=${LABEL_SMOOTHING} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=6_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=3_000 \
  trainer.max_steps=30_000 \
  wandb.group=train_classifier \
  wandb.name="ten_species-fudge_classifier" \
  hydra.run.dir="./outputs/ten_species/fudge_classifier"

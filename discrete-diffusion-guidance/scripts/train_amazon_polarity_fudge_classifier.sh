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
  --job-name=train_amazon_polarity_fudge_classifier \
  train_qm9_fudge_classifier.sh
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
  data=amazon_polarity \
  data.wrap=False \
  data.tokenizer_name_or_path=bert-base-uncased \
  data.label_col=label \
  data.num_classes=2 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=1024 \
  classifier_model=tiny-classifier \
  classifier_backbone=dit \
  classifier_model.pooling=no_pooling \
  model.length=128 \
  optim.lr=3e-4 \
  training.guidance=null \
  +training.use_label_smoothing=${LABEL_SMOOTHING} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=40_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.max_steps=-1 \
  +trainer.max_epochs=60 \
  trainer.val_check_interval=1.0 \
  wandb.group=train_classifier \
  wandb.name="amazon_polarity-fudge_classifier" \
  hydra.run.dir="./outputs/amazon_polarity/fudge_classifier"

#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
PROP=<qed|ring_count>
GUIDANCE=<cfg|fudge|cbg|pplm|nos>
... additional args for each guidance method ...
sbatch \
  --export=ALL,MODEL=${MODEL},PROP=${PROP},GUIDANCE=${GUIDANCE},... \
  --job-name=eval_qm9_${GUIDANCE}_${PROP}_${MODEL} \
  eval_qm9_guidance.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (choices: ar, mdlm, udlm)
#  - PROP (choices: qed, ring_count)
#  - GUIDANCE (each method has its own required args)
#  - CONDITION (optional: default = 1)
#  - SAMPLING_STEPS (optional: default = 32)
#  - SEED (optional: default = 1)

if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
if [ -z "${GUIDANCE}" ]; then
  echo "GUIDANCE is not set"
  exit 1
fi
if [ -z "${CONDITION}" ]; then
  CONDITION=1
fi
if [ -z "${SAMPLING_STEPS}" ]; then
  SAMPLING_STEPS=32
fi
if [ -z "${SEED}" ]; then
  SEED=1
fi


# CKPT below is unconditional model (will be overridden if GUIDANCE = "cfg")
if [ "${MODEL}"  = "ar" ]; then
  parameterization="ar"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=False
  SAMPLING_STEPS=32
  CKPT="${PWD}/outputs/qm9/ar_no-guidance"
elif [ "${MODEL}" = "mdlm" ]; then
  parameterization="subs"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=True
  CKPT="${PWD}/outputs/qm9/mdlm_no-guidance"
elif [ "${MODEL}" = "udlm" ]; then
  parameterization="d3pm"
  diffusion="uniform"
  TRAIN_T=0
  time_conditioning=True
  sampling_use_cache=False
  CKPT="${PWD}/outputs/qm9/udlm_no-guidance"
else
  echo "Invalid MODEL: ${MODEL}"
  exit 1
fi


guidance_args="guidance=${GUIDANCE} guidance.condition=${CONDITION}"
###### CFG ######
if [ "${GUIDANCE}" == "cfg" ]; then
  # Expecting:
  #  - GAMMA
  if [ -z "${GAMMA}" ]; then
    echo "GAMMA is not set"
    exit 1
  fi
  if [ "${PROP}" = "qed" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CKPT="${PWD}/outputs/qm9/ar_qed"
    elif [ "${MODEL}" = "mdlm" ]; then
      CKPT="${PWD}/outputs/qm9/mdlm_qed"
    elif [ "${MODEL}" = "udlm" ]; then
      CKPT="${PWD}/outputs/qm9/udlm_qed"
    fi
  elif [ "${PROP}" = "ring_count" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CKPT="${PWD}/outputs/qm9/ar_ring_count"
    elif [ "${MODEL}" = "mdlm" ]; then
      CKPT="${PWD}/outputs/qm9/mdlm_ring_count"
    elif [ "${MODEL}" = "udlm" ]; then
      CKPT="${PWD}/outputs/qm9/udlm_ring_count"
    fi
  else
    echo "Invalid PROP: ${PROP}"
    exit 1
  fi
  guidance_args="${guidance_args} guidance.gamma=${GAMMA}"
  results_csv_path="${CKPT}/qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.csv"
  generated_seqs_path="${CKPT}/samples-qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.json"
###### FUDGE / CBG ######
elif [ "${GUIDANCE}" = "fudge" ] || [ "${GUIDANCE}" = "cbg" ]; then
  # Expecting:
  #  - GAMMA
  #  - USE_APPROX (for cbg)
  if [ -z "${GAMMA}" ]; then
    echo "GAMMA is not set"
    exit 1
  fi
  if [ "${PROP}" = "qed" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/fudge_classifier/qed"
    elif [ "${MODEL}" = "mdlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/classifier/qed_absorbing_state_T-0"
    elif [ "${MODEL}" = "udlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/classifier/qed_uniform_T-0"
    fi
  elif [ "${PROP}" = "ring_count" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/fudge_classifier/ring_count"
    elif [ "${MODEL}" = "mdlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/classifier/ring_count_absorbing_state_T-0"
    elif [ "${MODEL}" = "udlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/classifier/ring_count_uniform_T-0"
    fi
  else
    echo "Invalid PROP: ${PROP}"
    exit 1
  fi
  guidance_args="${guidance_args} classifier_model=tiny-classifier classifier_backbone=dit guidance.classifier_checkpoint_path=${CLASS_CKPT}/checkpoints/best.ckpt guidance.gamma=${GAMMA}"
  if [ "${GUIDANCE}" = "fudge" ]; then
    guidance_args="${guidance_args} guidance.topk=40 classifier_model.pooling=no_pooling"  # Use full vocab size for topk
  fi
  if [ "${GUIDANCE}" = "cbg" ]; then
    if [ -z "${USE_APPROX}" ]; then
      echo "USE_APPROX is not set"
      exit 1
    fi
    guidance_args="${guidance_args} guidance.use_approx=${USE_APPROX}"
    results_csv_path="${CKPT}/qm9-eval-${GUIDANCE}_approx-${USE_APPROX}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.csv"
    generated_seqs_path="${CKPT}/samples-qm9-eval-${GUIDANCE}_approx-${USE_APPROX}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.json"
  else
    results_csv_path="${CKPT}/qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.csv"
    generated_seqs_path="${CKPT}/samples-qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_gamma-${GAMMA}_seed-${SEED}.json"
  fi
###### PPLM / NOS ######
elif [ "${GUIDANCE}" = "pplm" ] || [ "${GUIDANCE}" = "nos" ]; then
  if [ "${GUIDANCE}" = "pplm" ]; then
    # Expecting:
    #  - NUM_PPLM_STEPS
    #  - PPLM_STEP_SIZE
    #  - PPLM_STABILITY_COEF
    if [ -z "${NUM_PPLM_STEPS}" ]; then
      echo "NUM_PPLM_STEPS is not set"
      exit 1
    fi
    if [ -z "${PPLM_STEP_SIZE}" ]; then
      echo "PPLM_STEP_SIZE is not set"
      exit 1
    fi
    if [ -z "${PPLM_STABILITY_COEF}" ]; then
      echo "PPLM_STABILITY_COEF is not set"
      exit 1
    fi
    guidance_args="${guidance_args} guidance.num_pplm_steps=${NUM_PPLM_STEPS} guidance.pplm_step_size=${PPLM_STEP_SIZE} guidance.pplm_stability_coef=${PPLM_STABILITY_COEF}"
    results_csv_path="${CKPT}/qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_NUM_PPLM_STEPS-${NUM_PPLM_STEPS}_PPLM_STEP_SIZE-${PPLM_STEP_SIZE}_PPLM_STABILITY_COEF-${PPLM_STABILITY_COEF}_seed-${SEED}.csv"
    generated_seqs_path="${CKPT}/samples_qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_NUM_PPLM_STEPS-${NUM_PPLM_STEPS}_PPLM_STEP_SIZE-${PPLM_STEP_SIZE}_PPLM_STABILITY_COEF-${PPLM_STABILITY_COEF}_seed-${SEED}.json"
  else
    # Expecting:
    #  - NUM_NOS_STEPS
    #  - NOS_STEP_SIZE
    #  - NOS_STABILITY_COEF
    if [ -z "${NUM_NOS_STEPS}" ]; then
      echo "NUM_NOS_STEPS is not set"
      exit 1
    fi
    if [ -z "${NOS_STEP_SIZE}" ]; then
      echo "NOS_STEP_SIZE is not set"
      exit 1
    fi
    if [ -z "${NOS_STABILITY_COEF}" ]; then
      echo "NOS_STABILITY_COEF is not set"
      exit 1
    fi
    guidance_args="${guidance_args} guidance.num_nos_steps=${NUM_NOS_STEPS} guidance.nos_step_size=${NOS_STEP_SIZE} guidance.nos_stability_coef=${NOS_STABILITY_COEF}"
    results_csv_path="${CKPT}/qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_NUM_NOS_STEPS-${NUM_NOS_STEPS}_NOS_STEP_SIZE-${NOS_STEP_SIZE}_NOS_STABILITY_COEF-${NOS_STABILITY_COEF}_seed-${SEED}.csv"
    generated_seqs_path="${CKPT}/samples_qm9-eval-${GUIDANCE}_${PROP}_T-${SAMPLING_STEPS}_NUM_NOS_STEPS-${NUM_NOS_STEPS}_NOS_STEP_SIZE-${NOS_STEP_SIZE}_NOS_STABILITY_COEF-${NOS_STABILITY_COEF}_seed-${SEED}.json"
  fi

  if [ "${PROP}" = "qed" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/qed_ar"
    elif [ "${MODEL}" = "mdlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/qed_mdlm"
    elif [ "${MODEL}" = "udlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/qed_udlm"
    fi
  elif [ "${PROP}" = "ring_count" ]; then
    if [ "${MODEL}" = "ar" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/ring_count_ar"
    elif [ "${MODEL}" = "mdlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/ring_count_mdlm"
    elif [ "${MODEL}" = "udlm" ]; then
      CLASS_CKPT="${PWD}/outputs/qm9/pplm_classifier/ring_count_udlm"
    fi
  else
    echo "Invalid PROP: ${PROP}"
    exit 1
  fi
  guidance_args="${guidance_args} classifier_model=small-classifier classifier_backbone=dit guidance.classifier_checkpoint_path=${CLASS_CKPT}/checkpoints/best.ckpt"
else
  echo "Invalid GUIDANCE: ${GUIDANCE}"
  exit 1
fi

# shellcheck disable=SC2086
python -u guidance_eval/qm9_eval.py \
    hydra.output_subdir=null \
    hydra.run.dir="${CKPT}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode=qm9_eval \
    eval.checkpoint_path="${CKPT}/checkpoints/best.ckpt" \
    data=qm9 \
    data.label_col="${PROP}" \
    data.label_col_pctile=90 \
    data.num_classes=2 \
    model=small \
    backbone=dit \
    model.length=32 \
    training.guidance=null \
    parameterization=${parameterization} \
    diffusion=${diffusion} \
    time_conditioning=${time_conditioning} \
    T=${TRAIN_T} \
    sampling.num_sample_batches=64 \
    sampling.batch_size=16 \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.use_cache=${sampling_use_cache} \
    +eval.results_csv_path=${results_csv_path} \
    eval.generated_samples_path=${generated_seqs_path} \
    ${guidance_args}

#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate discdiff
 elif [[ "${CONDA_PREFIX}" != *"/discdiff" ]]; then
  conda deactivate
  conda activate discdiff
fi

# Setup HF cache
# shellcheck disable=SC1091
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"


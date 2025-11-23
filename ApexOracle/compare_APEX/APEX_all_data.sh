#!/bin/bash
#SBATCH --job-name=APEX_all_data             # 作业名称
#SBATCH --output=/home/otras/ors/fwa/tianang/projects/bert-loves-chemistry/logs/APEX_all.out           # 标准输出文件
#SBATCH --error=/home/otras/ors/fwa/tianang/projects/bert-loves-chemistry/logs/APEX_all.err            # 错误输出文件
#SBATCH --nodes=1                     # 使用的节点数
#SBATCH --ntasks=1                    # 使用的任务数
#SBATCH --cpus-per-task=32             # 每个任务使用的CPU核心数
#SBATCH --gres=gpu:a100:1                  # 使用的GPU数
#SBATCH --time=6:00:00               # 最大运行时间 (hh:mm:ss)
#SBATCH --mem=64GB                     # 内存需求

# 激活conda环境（如需要）
module load cesga/system miniconda3/22.11.1-1
conda activate ChemBERTa
wandb login 5518c5d9d8cbd8bc6edc7fbe88f6d5ab4e2832f9
cd /home/otras/ors/fwa/tianang/projects/bert-loves-chemistry/compare_APEX
# 执行具体命令
python APEX_train_DBAASP_MIC_5_fold_mean.py
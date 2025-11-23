#!/bin/bash
#SBATCH --job-name=002             # 作业名称
#SBATCH --output=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/002.out           # 标准输出文件
#SBATCH --error=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/002.err            # 错误输出文件
#SBATCH --partition=short           # 分区名称
#SBATCH --nodes=1                     # 使用的节点数
#SBATCH --ntasks=1                    # 使用的任务数
#SBATCH --cpus-per-task=32             # 每个任务使用的CPU核心数
#SBATCH --time=3-00:00:00               # 最大运行时间 (hh:mm:ss)
#SBATCH --mem=32GB                     # 内存需求

# 激活conda环境（如需要）
# source /mnt/lustre/scratch/nlsas/home/otras/ors/fwa/miniconda3/bin/activate
module load cesga/system miniconda3/22.11.1-1
conda activate
cd $LUSTRE/projects/Synergy/DataPrepare/MDLM
# 执行具体命令
python tokenize_SELFIES_descriptors_hf.py -s "002"
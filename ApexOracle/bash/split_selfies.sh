#!/bin/bash
#SBATCH --job-name=split_selfies             # 作业名称
#SBATCH --output=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/split_selfies.out           # 标准输出文件
#SBATCH --error=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/split_selfies.err            # 错误输出文件
#SBATCH --partition=short           # 分区名称
#SBATCH --nodes=1                     # 使用的节点数
#SBATCH --ntasks=1                    # 使用的任务数
#SBATCH --cpus-per-task=32             # 每个任务使用的CPU核心数
#SBATCH --time=2:00:00               # 最大运行时间 (hh:mm:ss)
#SBATCH --mem=128GB                     # 内存需求

# 激活conda环境（如需要）
# source /mnt/lustre/scratch/nlsas/home/otras/ors/fwa/miniconda3/bin/activate
module load cesga/system miniconda3/22.11.1-1
conda activate
cd $LUSTRE/projects/Synergy/DataPrepare/MDLM
# 执行具体命令
python split_selfies_csv_file.py
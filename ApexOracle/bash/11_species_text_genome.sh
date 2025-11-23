#!/bin/bash
#SBATCH --job-name=tg_11_3             # 作业名称
#SBATCH --output=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/tg_11_3.out           # 标准输出文件
#SBATCH --error=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/tg_11_3.err            # 错误输出文件
#SBATCH --partition=medium           # 分区名称
#SBATCH --nodes=1                     # 使用的节点数
#SBATCH --ntasks=1                    # 使用的任务数
#SBATCH --cpus-per-task=32             # 每个任务使用的CPU核心数
#SBATCH --gres=gpu:a100:1                  # 使用的GPU数
#SBATCH --time=1-12:00:00               # 最大运行时间 (hh:mm:ss)
#SBATCH --mem=64GB                     # 内存需求

# 激活conda环境（如需要）
# source /mnt/lustre/scratch/nlsas/home/otras/ors/fwa/miniconda3/bin/activate
module load cesga/system miniconda3/22.11.1-1
conda activate cold_base
cd $LUSTRE/projects/Synergy
# 执行具体命令
python -u MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble.py -p -t 3 -d 0 -e 25
#!/bin/bash
#SBATCH --job-name=t2_f2             # 作业名称
#SBATCH --output=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/3-strain-compare/t2_f2.out           # 标准输出文件
#SBATCH --error=/mnt/lustre/scratch/nlsas//home/otras/ors/tle/projects/logs/3-strain-compare/t2_f2.err            # 错误输出文件
#SBATCH --partition=long           # 分区名称
#SBATCH --nodes=1                     # 使用的节点数
#SBATCH --ntasks=1                    # 使用的任务数
#SBATCH --cpus-per-task=32             # 每个任务使用的CPU核心数
#SBATCH --gres=gpu:a100:1                  # 使用的GPU数
#SBATCH --time=4-00:00:00               # 最大运行时间 (hh:mm:ss)
#SBATCH --mem=64GB                     # 内存需求

# 激活conda环境（如需要）
# source /mnt/lustre/scratch/nlsas/home/otras/ors/fwa/miniconda3/bin/activate
module load cesga/system miniconda3/22.11.1-1
conda activate cold_base
cd $LUSTRE/projects/Synergy
# 执行具体命令
python -u antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad.py -p -t 2 -d 0 -f 2 -e 25

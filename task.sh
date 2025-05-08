#!/bin/bash
#SBATCH -J neat_vollyball           # 作业的名称 可根据需要自行命名
#SBATCH -p ihictest  
#SBATCH -N 1                  # 申请的节点数1个
#SBATCH --cpus-per-task=8

# 设置 OpenMP 使用的线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 设置 multiprocessing 的启动方法为 spawn
export PYTHONMULTIPROCESSING_START_METHOD=spawn

WDIR=`pwd`   #获取当前目录
cd "$WDIR"/neat-volleyball || exit

eval "$(conda shell.bash hook)"
conda activate ai

python train.py -p 200 -g 50 --sample_num 10

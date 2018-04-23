#!/bin/bash 
#SBATCH --ntasks=1              # four processes
#SBATCH --job-name=dorefa
#SBATCH --time=48:00:00
#SBATCH --array=1-4
#SBATCH --partition=sm

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2gpu/bin/activate
cd /data/sls/u/meng/skanda/home/thesis/dorefa/real

MODELS=(buffer fcn1 fcn2 cnn lcn)

# srun --gres=gpu:1 --partition=sm -N1-1 sharegpu_run.sh ${MODELS[$SLURM_ARRAY_TASK_ID]}
srun --gres=gpu:1 sharegpu_run.sh ${MODELS[$SLURM_ARRAY_TASK_ID]}

wait


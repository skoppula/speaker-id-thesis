#!/bin/bash 
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --job-name=prDF16qeF
#SBATCH --time=4:00:00
#SBATCH --array=1-4
#SBATCH --partition=cpu

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2cpu/bin/activate
cd /data/sls/u/meng/skanda/home/thesis/dorefa/real

MODELS=(buffer lcn fcn1 fcn2 cnn)
model=${MODELS[$SLURM_ARRAY_TASK_ID]}
quant_ends=False
btw=16
bita=32
dorefa=True

srun --cpus-per-task=4 --time=1:00:00 python prune.py --model=$model --bitw=$btw --bita=$bita --quant_ends=${quant_ends} --dorefa=$dorefa &

wait


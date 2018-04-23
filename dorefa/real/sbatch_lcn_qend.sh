#!/bin/bash 
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --job-name=lcn_quant
#SBATCH --time=52:00:00
#SBATCH --array=1-5

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2gpu/bin/activate
cd /data/sls/u/meng/skanda/home/thesis/dorefa/real

model=lcn
BITW=(buffer 2 4 8 16 32)
btw=${BITW[$SLURM_ARRAY_TASK_ID]}
quant_ends=True
bita=32
loaddir="train_log/lcn_w_${btw}_a_32_quant_ends_False/checkpoint"

srun --cpus-per-task=4 --gres=gpu:1 --time=48:00:00 python drf_run.py --model_name=$model --bitw=$btw --bita=$bita --quant_ends=${quant_ends} --load_ckpt=$loaddir &

wait


#!/bin/bash 
#SBATCH --cpus-per-task=8      # four cores per process
#SBATCH --ntasks=1             # four processes
#SBATCH --job-name=bit32linear
#SBATCH --time=2:00:00
#SBATCH --array=1-3
#SBATCH --partition=cpu

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2cpu/bin/activate
cd /data/sls/r/u/skanda/home/thesis/manfxpt

MODELS=(buffer fcn2 cnn lcn)
w_bits=4
a_bits=8
bias_bits=32

reg=True
load_ckpt="/data/sls/u/meng/skanda/home/thesis/dorefa/no_bn_models/${MODELS[$SLURM_ARRAY_TASK_ID]}/checkpoint"

srun --cpus-per-task=8 --time=2:00:00 --partition=cpu python drf_run.py --model=${MODELS[$SLURM_ARRAY_TASK_ID]} --load_ckpt=$load_ckpt --a_bits=$a_bits --w_bits=$w_bits --bias_bits=$bias_bits &

wait


#!/bin/bash 
#SBATCH --gres=gpu:1           # grab 2 gpus 
#SBATCH --cpus-per-task=8      # four cores per process
#SBATCH --ntasks=1             # four processes
#SBATCH --job-name=sre-dnn
#SBATCH --time=52:00:00
#SBATCH --array=1-2

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2gpu/bin/activate
cd /data/sls/r/u/skanda/home/thesis/baseexps/sre

# MODELS=(buffer fcn1 fcn2 cnn lcn maxout1 maxout2 dsc1 dsc2)
# MODELS=(buffer maxout1 maxout2 dsc1 dsc2)
MODELS=(buffer dsc1 dsc2)

# srun --cpus-per-task=8 --gres=gpu:1 --time=48:00:00 python rsr-run.py --model=${MODELS[$SLURM_ARRAY_TASK_ID]} --bn=False --reg=False --context=50 &
# srun --cpus-per-task=8 --gres=gpu:1 --time=48:00:00 python rsr-run.py --model=${MODELS[$SLURM_ARRAY_TASK_ID]} --bn=False --reg=True --context=50 &
# srun --cpus-per-task=8 --gres=gpu:1 --time=48:00:00 python rsr-run.py --model=${MODELS[$SLURM_ARRAY_TASK_ID]} --bn=True --reg=False --context=50 &
srun --cpus-per-task=8 --gres=gpu:1 --time=48:00:00 python sre-run.py --model=${MODELS[$SLURM_ARRAY_TASK_ID]} --bn=True --reg=False --context=50 --notes=noLRSchedule &

wait


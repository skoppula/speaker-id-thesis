#!/bin/bash 
#SBATCH --gres=gpu:1           # grab 2 gpus 
#SBATCH --cpus-per-task=8      # four cores per process
#SBATCH --ntasks=1             # four processes
#SBATCH --job-name=spkid_adam
#SBATCH --time=52:00:00
#SBATCH --array=1-4

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
source /data/sls/r/u/skanda/home/envs/tf2gpu/bin/activate
cd /data/sls/r/u/skanda/home/thesis/baseexps

basedir=/data/sls/r/u/skanda/home/thesis/baseexps/train_log
python rsr-run.py --model=dsc1 --notes=noLRSchedule --bn=True --reg=False --context=50 --load_ckpt=${basedir}/sentfiltNone_dsc1_bnTrue_regFalse_noLRSchedule/checkpoint & \
python rsr-run.py --model=dsc1 --notes=noLRSchedule --bn=True --reg=True --context=50 --load_ckpt=${basedir}/sentfiltNone_dsc1_bnTrue_regTrue_noLRSchedule/checkpoint & \
python rsr-run.py --model=dsc2 --notes=noLRSchedule --bn=True --reg=False --context=50 --load_ckpt=${basedir}/sentfiltNone_dsc2_bnTrue_regFalse_noLRSchedule/checkpoint & \
python rsr-run.py --model=dsc2 --notes=noLRSchedule --bn=True --reg=True --context=50 --load_ckpt=${basedir}/sentfiltNone_dsc2_bnTrue_regTrue_noLRSchedule/checkpoint & \

python rsr-run.py --model=maxout1 --notes=noLRSchedule --bn=True --reg=False --context=50 --load_ckpt=${basedir}/sentfiltNone_maxout1_bnTrue_regFalse_noLRSchedule/checkpoint & \
python rsr-run.py --model=maxout2 --notes=noLRSchedule --bn=True --reg=False --context=50 --load_ckpt=${basedir}/sentfiltNone_maxout2_bnTrue_regFalse_noLRSchedule/checkpoint & \

python rsr-run.py --model=dsc1 --notes=noLRSchedule --bn=False --reg=False --context=50 & \
python rsr-run.py --model=dsc2 --notes=noLRSchedule --bn=False --reg=False --context=50 & \
python rsr-run.py --model=maxout1 --notes=noLRSchedule --bn=False --reg=False --context=50 & \
python rsr-run.py --model=maxout2 --notes=noLRSchedule --bn=False --reg=False --context=50 & \
wait


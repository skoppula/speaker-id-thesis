#!/bin/bash 
# srun --cpus-per-task=16 --partition=sm --gres=gpu:1 -N1-1 --time=48:00:00  -o sre_dsc_redo.out -e sre_dsc_redo.err  --job-name=sre_dsc_redo srun.sh &

echo "$(hostname) $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
source /data/sls/r/u/skanda/home/envs/tf2gpu/bin/activate
cd /data/sls/r/u/skanda/home/thesis/baseexps/sre

basedir=/data/sls/r/u/skanda/home/thesis/baseexps/sre/train_log
python sre-run.py --model=dsc1 --bn=True --reg=False --context=50 --notes=noLRSchedule --load_ckpt=${basedir}/sentfiltNone_dsc1_bnTrue_regFalse_noLRSchedule/checkpoint & \
python sre-run.py --model=dsc1 --bn=True --reg=True --context=50 --notes=noLRSchedule --load_ckpt=${basedir}/sentfiltNone_dsc1_bnTrue_regFalse_noLRSchedule/checkpoint & \
python sre-run.py --model=dsc2 --bn=True --reg=False --context=50 --notes=noLRSchedule & \
python sre-run.py --model=dsc2 --bn=True --reg=True --context=50 --notes=noLRSchedule --load_ckpt=${basedir}/sentfiltNone_dsc2_bnTrue_regTrue_noLRSchedule/checkpoint & \

wait


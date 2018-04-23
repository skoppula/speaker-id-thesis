#! /bin/bash

model=$1
bita=32
quant_ends="False"
cd /data/sls/u/meng/skanda/home/thesis/dorefa/real
echo $model
# srun --gres=gpu:1 --partition=sm -N1-1 sharegpu_run.sh fcn1

python drf_run.py --model_name=$model --bitw=32 --bita=$bita --quant_ends=${quant_ends} & \
python drf_run.py --model_name=$model --bitw=16 --bita=$bita --quant_ends=${quant_ends} & \
python drf_run.py --model_name=$model --bitw=8 --bita=$bita --quant_ends=${quant_ends} & \
python drf_run.py --model_name=$model --bitw=4 --bita=$bita --quant_ends=${quant_ends} & \
python drf_run.py --model_name=$model --bitw=2 --bita=$bita --quant_ends=${quant_ends} &

wait


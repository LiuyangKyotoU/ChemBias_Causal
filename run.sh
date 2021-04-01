#!/bin/bash

echo $1
echo $2

CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 0
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 1
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 2
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 3
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 4
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 5
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 6
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 7
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 8
CUDA_VISIBLE_DEVICES=$1 python train.py --task $2 --trial 9
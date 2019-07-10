#!/bin/bash

#SBATCH -p lyceum

#SBATCH --time=60:00:00          # walltime
#SBATCH --gres=gpu:1

GPU_ID=0          
CUDA_VISIBLE_DEVICES=$GPU_ID

MODEL='resnet152'

python ../../keras_retinanet/bin/train.py --gpu $CUDA_VISIBLE_DEVICES --batch-size 1 --steps 10000 --backbone $MODEL --epochs 10 \
									--compute-val-loss --freeze-backbone \
									--snapshot-path /mainfs/lyceum/ncdt1u18/keras-retinanet/results/$MODEL/frozen \
									--tensorboard-dir /mainfs/lyceum/ncdt1u18/keras-retinanet/results/$MODEL/logs \
									--weights /mainfs/lyceum/ncdt1u18/keras-retinanet/imagenet-weights/ResNet-152-model.keras.h5\
									csv /mainfs/lyceum/ncdt1u18/data/acfr-fruit-dataset/apples/rectangular_annotations/train_annotations.csv \
									/mainfs/lyceum/ncdt1u18/data/acfr-fruit-dataset/apples/rectangular_annotations/classes.csv \
									--val-annotations /mainfs/lyceum/ncdt1u18/data/acfr-fruit-dataset/apples/rectangular_annotations/val_annotations.csv

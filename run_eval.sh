#!/bin/bash

#SBATCH -p lyceum

#SBATCH --time=60:00:00          # walltime
#SBATCH --gres=gpu:1
 
python ./keras_retinanet/bin/evaluate.py  --backbone 'resnet152' --iou-threshold 0.5 --convert-model --save-path ./dummy_predictions --score-threshold 0.99 \
 											csv ./data/acfr-fruit-dataset/apples/rectangular_annotations/dummy_val_annotations.csv ./data/acfr-fruit-dataset/apples/rectangular_annotations/classes.csv \
 											./resnet152_csv_10.h5
 											
 										

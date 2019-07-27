#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import numpy as np
import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
            
def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator

def compute_stats(average_precisions, pr_curves, iou, generator):
        """ Perform the computations for the mAP, mF1-score and the rest of the metrics.
            Args
                average_precisions : The average precisions coming from utils/eval.py.
                pr_curves          : The PR curves coming from utils/eval.py.
                iou                : The IoU threshold that these metrics were computed at.
                generator          : The samples generator.
            Returns 
                the mAP mF1-score and other metrics.
            """
        total_instances = []
        precisions = []
        f1_scores = []
        mean_ious = []
        print('\nIoU@{}'.format(iou))

        for label, (average_precision, num_annotations) in average_precisions.items():
            #print('{:.0f} instances of class'.format(num_annotations),
            #      generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            f1_scores.append(max(pr_curves[label]['f1_score']))
            mean_ious.append(np.mean(pr_curves[label]['average_iou']))

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        

        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        mean_f1 = sum(f1_scores) / sum(x > 0 for x in total_instances)
        mean_iou = sum(mean_ious) / sum(x > 0 for x in total_instances)

        for label in range(generator.num_classes()):
            class_label     = generator.label_to_name(label)
            instances       = int(total_instances[label])
            predictions     = len(pr_curves[label]['precision'])
            true_positives  = int(pr_curves[label]['TP'][-1]) if len(pr_curves[label]['TP']) > 0 else 0
            false_positives = int(pr_curves[label]['FP'][-1]) if len(pr_curves[label]['FP']) > 0 else 0

            print('Class {}: Instances: {} | Predictions: {} | False positives: {} | True positives: {}'.format(
                    class_label, instances, predictions, false_positives, true_positives))

        return mean_ap, mean_f1, mean_iou

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--logs',             help='Path to save statistics like the PR curves.')

    #NMS arguments
    parser.add_argument('--nms-threshold', help='Threshold for the IoU value to determine when a box should be suppressed.', dest='nms_threshold', type=float, default=0.5)
    parser.add_argument('--nms-score', help='Threshold used to prefilter the boxes with.', dest='nms_score', type=float, default=0.05)
    parser.add_argument('--nms-detections', help='Maximum number of detections to keep.', dest='nms_detections', type=int, default=300)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(
            model = model,
            nms_threshold = args.nms_threshold,
            score_threshold = args.nms_score,
            max_detections = args.nms_detections, 
            anchor_params=anchor_params
            )

    # print model summary
    # print(model.summary())

    # start evaluation
    iou_thresholds = [0.2, 0.5, 0.75]
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        for iou in iou_thresholds:
            average_precisions, pr_curves = evaluate(
                generator,
                model,
                iou_threshold=iou,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                save_path=args.save_path,
                verbose=0
            )

            mean_ap, mean_f1, mean_iou = compute_stats(average_precisions, pr_curves, iou, generator)

        for iou in iou_thresholds:
            print('IoU@{} =>'.format(iou),  'mAP: {:.4f}'.format(mean_ap), 'mF1-score: {:.4f}'.format(mean_f1), 'mIoU: {:.4f}'.format(mean_iou))

        # save stats
        if args.logs:
            makedirs(args.logs)
            np.save(os.path.join(args.logs, 'pr_curves'), pr_curves)
            
if __name__ == '__main__':
    main()

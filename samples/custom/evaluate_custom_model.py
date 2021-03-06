# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:54:25 2018

@author: hyungu
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import json
import random
import datetime
import collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project (The Mask-RCNN directory)
try:
    # Using PyCharm's interactive console from project-level directory
    ROOT_DIR = os.path.abspath('./')
    assert os.path.isdir(os.path.join(ROOT_DIR, 'mrcnn'))
except:
    ROOT_DIR = os.path.abspath('../../')  # Running 'train_custom_model.py'
    assert os.path.isdir(os.path.join(ROOT_DIR, 'mrcnn'))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from mrcnn import model as modellib
from samples.custom.custom import CustomConfig, CustomDataset

# Directory to dataset (right above train/val)
try:
    # Using PyCharm's interactive console from project-level directory
    DATASET_DIR = os.path.abspath('../dataset/')
    assert os.path.isdir(DATASET_DIR)
except:
    # Running 'evaluate_custom_model.py'
    DATASET_DIR = os.path.abspath('../../../dataset/')
    assert os.path.isdir(DATASET_DIR)

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR = DEFAULT_LOGS_DIR
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_custom_20180930T1453.h5')
assert os.path.exists(CUSTOM_MODEL_PATH)

# Configure device
DEVICE = '/gpu:0'

# Inspection mode
TEST_MODE = 'inference'
assert TEST_MODE in ['training', 'inference']


class CustomInferenceConfig(CustomConfig):
    '''Inference configuration class.'''
    
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # MISC
    IMAGE_RESIZE_MODE = 'square'
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    # Skip detections with under minimum confidence
    DETECTION_MIN_CONFIDENCE = 0.8


def get_ax(rows=1, cols=1, size=16):
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig, ax


if __name__ == '__main__':
    
    # Hyperparameter configuration for inference
    inf_config = CustomInferenceConfig()
    inf_config.display()
    
    # Load validation dataset
    subset = 'test'
    dataset_val = CustomDataset()
    dataset_val.load_custom_data(dataset_dir=DATASET_DIR,
                                 subset='test')
    
    # Prepare for inference
    dataset_val.prepare()
    
    # Print basic dataset statistics
    print('>>> #. images: {}'.format(dataset_val.num_images))
    for class_info in dataset_val.class_info:
        print('>>> ID: {} | class name: {}'.format(
                class_info['id'], class_info['name']))
    
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode=TEST_MODE,
                                  model_dir=MODEL_DIR,
                                  config=inf_config)

    # Load trained weights        
    print('>>> Loading trained weights...'); start = time.time()
    model.load_weights(CUSTOM_MODEL_PATH, by_name=True)
    print('>>> Loaded trained weights successfully ({:.3f}s)...'.format(
            time.time() - start))

    now = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    writedir = os.path.join(MODEL_DIR, '{}_{}_{}'.format(
            subset, now, inf_config.DETECTION_MIN_CONFIDENCE))
    os.makedirs(writedir, exist_ok=True)


    iou_dict = collections.defaultdict(list)
    for image_id in dataset_val.image_ids:

        info = dataset_val.image_info[image_id]
        print('>>> Image ID: {} | SOURCE.ID: {}.{}'.format(
                image_id, info['source'], info['id']))
        img_ref = dataset_val.image_reference(image_id)
        print('>>> Image reference: {}'.format(
                img_ref))
    
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset=dataset_val,
                                   config=inf_config,
                                   image_id=image_id,
                                   use_mini_mask=False)
    
        # Run object detection
        results = model.detect([image], verbose=1)
        result = results[0]

        gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
                                                               result['rois'], result['class_ids'],
                                                               result['scores'], result['masks'],
                                                               iou_threshold=0.5, score_threshold=0.0)

        iou_values = []
        for i, j in enumerate(pred_match):
            if j > -1.:
                j = int(j)
                iou = overlaps[i, j]
                iou_values.append(iou)
                class_id = gt_class_id[j]
                iou_dict[class_id].append(iou)
            else:
                continue

        if len(iou_values) < 1:
            miou = 0.
        else:
            miou = np.mean(iou_values)


        # Display results
        fig, ax = get_ax(1)
        visualize.display_instances(image=image,
                                    boxes=result['rois'],
                                    masks=result['masks'],
                                    class_ids=result['class_ids'],
                                    class_names=dataset_val.class_names,
                                    scores=result['scores'],
                                    ax=ax,
                                    title='(mIOU:{:.3f})Predictions (ID:{})\n{}'.format(
                                            miou, image_id, img_ref)
                                    )
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        
        # Save to file
        writefile = os.path.join(writedir,
                                 info['id'].split('.')[0] + '_{:.3f}.png'.format(
                                         miou))
        fig.savefig(fname=writefile)
        plt.close(fig)

    # Print iou dictionary to json file
    for class_id, iou_list in iou_dict.items():
        print('>>> {}: {:.3f}'.format(class_id, np.mean(iou_list)))

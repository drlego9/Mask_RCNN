# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 02:03:36 2018

@author: hyungu
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys

import numpy as np
import tensorflow as tf

# Root directory of the project (The Mask-RCNN directory)
os.chdir('D:/projects/PROJECT_hyundai/2018/Mask_RCNN/samples/custom/')
ROOT_DIR = os.path.abspath('../../')

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn import model as modellib
from mrcnn import utils
from custom import CustomConfig, CustomDataset

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory to dataset (right above train/val)
DATASET_DIR = 'D:/projects/PROJECT_hyundai/2018/datasets/20180823/'

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = DEFAULT_LOGS_DIR


class CustomTrainConfig(CustomConfig):
    """Add class docstring."""
    
    # Give the configuration a recognizable name
    NAME = "custom"

    # Number of images to fit to a single GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 7  # background, my_road, other_road, car
                         # bump, manhole, pothole, steel
    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # MISC
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (28, 28)


if __name__ == '__main__':
    
    # GPU memory usage configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)  # Pass this session when training...
    
    # Train configuration
    train_config = CustomTrainConfig()
    DISPLAY = False
    if DISPLAY:
        train_config.display()
    
    # Prepare datasets (train & val)
    dataset_train = CustomDataset()
    dataset_train.load_custom_data(DATASET_DIR, 'train')
    dataset_train.prepare()
    
    dataset_val = CustomDataset()
    dataset_val.load_custom_data(DATASET_DIR, 'val')
    dataset_val.prepare()
    
    # Prepare Mask-RCNN model
    model = modellib.MaskRCNN(mode='training',
                              config=train_config,
                              model_dir=MODEL_DIR)
    
    # Initialize weights
    init_with = 'coco'
    print('>>> Loading weights with {}...'.format(init_with))
    if init_with == 'coco':
        model.load_weights(filepath=COCO_MODEL_PATH,
                           by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == 'last':
        model.load_weights(filepath=model.find_last(),
                           by_name=True)
    else:
        raise ValueError
        
    # Train, phase 1 (layers='heads')
    model.train(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=50,
                layers='heads')
    
    # Train, phase 2 (layers='all')
    fine_tune = False
    if fine_tune:
        model.train(dataset_train, dataset_val, 
                    learning_rate=train_config.LEARNING_RATE / 10,
                    epochs=50, 
                    layers="all")

    # Save trained model
    model_path = os.path.join(MODEL_DIR, 'mask_rcnn_custom.h5')
    model.keras_model.save_weights(model_path)
    
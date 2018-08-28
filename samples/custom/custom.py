#-*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import skimage
import operator
import numpy as np

# Root directory of the project (The Mask-RCNN directory)
os.chdir('D:/projects/PROJECT_hyundai/2018/Mask_RCNN/samples/custom/')
ROOT_DIR = os.path.abspath('../../')

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to dataset (right above train/val)
DATASET_DIR = 'D:/projects/PROJECT_hyundai/2018/datasets/20180823/'


class CustomConfig(Config):
    """Add class docstring."""

    # Give the configuration a recognizable name
    NAME = "custom"

    # Number of images to fit to a single GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 7  # background, pothole, manhole, steel,
                         # bump, car, my_road, other_road

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CustomDataset(utils.Dataset):
    """Add class docstring."""
    def __init__(self, class_map=None):
        super(CustomDataset, self).__init__(class_map=class_map)
        self.custom_class_names = ['BG', 'pothole', 'manhole', 'steel',
                                   'bump', 'car', 'my_road', 'other_road']
        
        for i, class_name in enumerate(self.custom_class_names):
            if i == 0 and class_name == 'BG':
                continue
            else:
                self.add_class(source='custom',
                               class_id=i,
                               class_name=class_name)

    def load_custom_data(self, dataset_dir, subset):
        """Load a subset of the Custom dataset.
        dataset_dir: str, root directory of the dataset.
        subset: str, subset name to load, one of 'train' or 'val'.
        """
        
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # subirs = ['pothole', 'manhole', 'bump', 'steel']
        subdirs = os.listdir(dataset_dir)
        subdirs = [os.path.join(dataset_dir, sd) for sd in subdirs]
        
        for subdir in subdirs:
            
            image_dirs = os.listdir(subdir)
            image_dirs = [os.path.join(subdir, imd) for imd in image_dirs]
            assert all([os.path.isdir(imd) for imd in image_dirs])
            
            for image_dir in image_dirs:
                
                # Load image
                image_path = os.path.join(image_dir, 'img.png')
                image_path = image_path.replace('\\', '/')  # Replace backslashes
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]  # image.shape == (1080, 1920, 3)
                
                # Load polygons
                polygon_path = os.path.join(image_dir, 'labelme_polygons.json')
                with open(polygon_path, 'r') as fp:
                    labelme_polygons = json.load(fp)
                    assert isinstance(labelme_polygons, dict)
                    
                # Make polygons (1 polygons for single image)
                polygons = []
                for p in labelme_polygons['shapes']:
                    
                    # Check if the label is of interest
                    if p['label'] not in self.custom_class_names:
                        continue
                    
                    polygon = {}
                    
                    # Make value for the following keys: ['all_points_x', 'all_points_y']
                    xy_pairs = p['points']
                    polygon['all_points_x'] = [xy[0] for xy in xy_pairs]
                    polygon['all_points_y'] = [xy[1] for xy in xy_pairs]
                    polygon['name'] = p['label']
                    
                    polygons.append(polygon)
                
                self.add_image(
                        source='custom',
                        image_id=labelme_polygons['imagePath'], # TODO: check sanity
                        path=image_path,
                        width=width,
                        height=height,
                        polygons=polygons)
                
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        
        # If not a custom image, delegate to parent class.
        info = self.image_info[image_id]
        assert isinstance(info, dict)
        if info['source'] != 'custom':
            return super(CustomDataset, self).load_mask(image_id)
        
        # Get class names for class info
        class_names = map(operator.itemgetter('name'), self.class_info)
        class_names = list(class_names)
        
        # Get polygons for the current image
        polygons = info['polygons']; assert isinstance(polygons, list)
        
        height = info['height']
        width = info['width']
        
        count = len(polygons)
        
        class_ids = []
        instance_masks = []
        for polygon in polygons:
            
            # Get class ids
            class_name = polygon['name']
            class_id = class_names.index(class_name)
            class_ids.append(class_id)
            
            # Get x-y coordinates to draw mask
            x_coords = polygon['all_points_x']
            y_coords = polygon['all_points_y']
            xy_pairs = np.asarray(list(zip(x_coords, y_coords)),
                                  dtype=np.int32)
            
            # Draw mask
            mask = np.zeros((height, width),
                            dtype=np.int32)
            mask = cv2.fillConvexPoly(mask, xy_pairs, int(True))
            instance_masks.append(mask)
            
        # Stack to 3D numpy array
        instance_masks = np.stack(instance_masks, axis=-1)
        
        # 'class_ids' as 1D numpy array
        class_ids = np.array(class_ids, dtype=np.int32)
        
        # Sort in importance order
        sort_idx = np.argsort(class_ids)
        class_ids = class_ids[sort_idx].copy()
        instance_masks = instance_masks[:, :, sort_idx].copy()
        
        # Handle mask occlusion
        occlusion = np.logical_not(instance_masks[:, :, 0]).astype(np.uint8)
        for i in range(1, count, 1):
            instance_masks[:, :, i] = instance_masks[:, :, i] * occlusion
            occlusion = np.logical_and(
                    occlusion, np.logical_not(instance_masks[:, :, i]))
            
        # Final assertion
        assert instance_masks.shape[-1] == class_ids.shape[0]
        return instance_masks.astype(np.int32), class_ids
        
            
    def image_reference(self, image_id):
        """Returns the absolute path to the image."""
        info = self.image_info[image_id]
        if info['source'] == 'custom':
            return info['path']
        else:
            super(CustomDataset, self).image_reference(image_id)

        
if __name__ == '__main__':
    
    print('>>> Root directory: {}'.format(ROOT_DIR))
    print('>>> Path to COCO model weight file: {}'.format(COCO_MODEL_PATH))
    print('>>> Path to default log directory: {}'.format(DEFAULT_LOGS_DIR))
    print('>>> Path to dataset directory: {}'.format(DATASET_DIR))

    dataset_train = CustomDataset()
    dataset_train.load_custom_data(DATASET_DIR, 'train')
    dataset_train.prepare()
    
    dataset_val = CustomDataset()
    dataset_val.load_custom_data(DATASET_DIR, 'val')
    dataset_val.prepare()
    
    print('>>> TRAIN DATA...')
    print('>>> #. of images: {}'.format(len(dataset_train.image_ids)))
    for class_id, class_name in zip(dataset_train.class_ids, dataset_train.class_names):
        print('>>> Class {}: {}'.format(class_id, class_name))
    
    print('>>> VALID DATA...')
    print('>>> #. of images: {}'.format(len(dataset_val.image_ids)))
    for class_id, class_name in zip(dataset_val.class_ids, dataset_val.class_names):
        print('>>> Class {}: {}'.format(class_id, class_name))
        
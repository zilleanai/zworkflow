import os
import numpy as np
import cv2
from PIL import Image
from .datasetbase import DataSetBase


class SegmentationDataset(DataSetBase):

    def __init__(self, config, preprocessing=None):
        super().__init__(config)
        self.features = config['dataset']['features']
        self.labels = config['dataset']['labels']
        self.width = config['dataset']['width']
        self.height = config['dataset']['height']
        self.preprocessing = preprocessing
        self.load(config['dataset']['train_images'], config['dataset']['train_masks'])

    def load(self, images='.', masks='.'):
        self.images = sorted([os.path.join(images,f) for f in os.listdir(images)
                             if f.endswith('.png') or f.endswith('.tif') or f.endswith('.jpg')])
        self.masks = sorted([os.path.join(masks,f) for f in os.listdir(masks)
                             if f.endswith('.png') or f.endswith('.tif') or f.endswith('.jpg')])
        assert(len(self.images) == len(self.masks))

    def load_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = np.array(image)
        return image
    
    def load_mask(self, path):
        mask = Image.open(path)
        mask = np.array(mask)
        return mask

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = self.load_image(image)
        mask = self.load_mask(mask)
        (image, mask) = self.preprocessing.process((image, mask))

        image = np.rollaxis(image, 2, 0)
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        return [image, mask]

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return "images: " + str(self.config['dataset']['train_images']) + " masks: " + str(self.config['dataset']['train_masks']) + " len: " + str(len(self.images))


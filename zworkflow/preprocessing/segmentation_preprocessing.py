
import cv2
from .preprocessingbase import PreprocessingBase


class SegmentationPreprocessing(PreprocessingBase):
    def __init__(self, config):
        super().__init__(config)
        self.functions['resize'] = self.resize
        self.functions['normalize'] = self.normalize
        self.functions['classify_mask'] = self.classify_mask

        self.width = self.config['dataset']['width']
        self.height = self.config['dataset']['height']

    def normalize(self, data):
        # source: https://pytorch.org/docs/stable/torchvision/models.html
        (img, mask) = data
        normed_img = ((img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
        return (normed_img, mask)

    def resize(self, data):
        (img, mask) = data
        img = cv2.resize(
            img, (self.height, self.width))
        if mask is None:
            return (img, None)
        mask = cv2.resize(
            mask, (self.height, self.width), interpolation=cv2.INTER_NEAREST)
        return (img, mask)

    def classify_mask(self, data):
        (img, mask) = data
        if mask is None:
            return data
        mask[mask != 255] = 0
        mask[mask == 255] = 1
        return (img, mask)

    def process(self, data, verbose=False):
        for f in self.config['preprocessing']['functions']:
            fun = self.functions[f]
            data = fun(data)
        return data

    def __str__(self):
        return "segmentation preprocessing: " + str(list(self.functions.keys()))


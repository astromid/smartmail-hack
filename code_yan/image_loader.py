import cv2
import numpy as np
from skimage.exposure import adjust_gamma
from base_dataloader import DataLoader
from task_config import RESCALE_SIZE
from task_config import CROP_SIZE
from keras.applications import *
from keras_contrib.applications import *

# known pretrained networks
CLF2MODULE = {
    'densenet40':   'densenet',
    'densenet121':  'densenet',
    'densenet169':  'densenet',
    'densenet201':  'densenet',
    'resnet50':     'resnet50',
    'xception':     'xception',
    'inception_resnet_v2': 'inception_resnet_v2',
    'ror':          'ror'
}
CLF2CLASS = {
    'densenet40':   'DenseNet40',
    'densenet121':  'DenseNet121',
    'densenet169':  'DenseNet169',
    'densenet201':  'DenseNet201',
    'resnet50':     'ResNet50',
    'xception':     'Xception',
    'inception_resnet_v2': 'InceptionResNetV2',
    'ror':          'ResidualOfResidual'
}


class ImageLoader(DataLoader):

    def __init__(self, crops, *args, **kwargs):
        self.crops = crops
        self.crop_mode = None
        super().__init__(*args, **kwargs)

        if self.clf_name in CLF2MODULE:
            print("Found known pretrained network")
            print(f"The preprocess function for {self.clf_name} is used")
            module_name = CLF2MODULE[self.clf_name]
            self._preprocess_batch = getattr(globals()[module_name], 'preprocess_input')
        else:
            print(f"Can\'t found suitable preprocess function for {self.clf_name}")

    def _load_sample(self, sample_path):
        # try to load all jpeg & png files with opencv
        image = cv2.imread(sample_path)
        if image is not None:
            shape = image.shape
            # normal 3-channel image
            if shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # grayscale image
            if len(shape) == 2 or shape[-1] == 1:
                image = image.reshape((shape[0], shape[1], 1))
                image = np.concatenate([image, image, image], axis=-1)
        return image

    def _prepare_sample(self, image):
        if self.crops:
            # pad image to CROP_SIZE
            # image = self._pad_image(image)
            # rescale instead of padding
            h, w, _ = image.shape
            if h < CROP_SIZE or w < CROP_SIZE:
                image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
            # use random crops while training & validation
            if self.mode != 'test':
                image = self._random_crop_image(image)
            # else took 5 crops
            # TODO: TTA instead of random crop
            else:
                image = self._random_crop_image(image)
        else:
            image = cv2.resize(image, (RESCALE_SIZE, RESCALE_SIZE))
        return image

    def _augment_sample(self, image):
        # TODO: color? may be it's time to use imgaug
        # zoom
        # if np.random.rand() < 0.5:
        #     scale = np.random.choice([1.2, 1.5])
        #     image = cv2.resize(image, None, scale, scale)
        # rotate 90-180-270
        # if np.random.rand() < 0.5:
        #     n = np.random.choice([1, 2, 3])
        #     for _ in range(n):
        #         image = np.rot90(image, 1, (0, 1))
        # flip
        # if np.random.rand() < 0.5:
        #     image = np.flip(image, 1)
        # gamma
        if np.random.rand() < 0.5:
            gamma = np.random.choice([0.5, 0.8, 1.2, 1.5])
            image = adjust_gamma(image, gamma)
        # blur
        if np.random.rand() < 0.5:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    @staticmethod
    def _random_crop_image(image):
        h, w, _ = image.shape
        h_start = np.random.randint(0, max(h - CROP_SIZE, 1))
        w_start = np.random.randint(0, max(w - CROP_SIZE, 1))
        return image[h_start:h_start + CROP_SIZE, w_start:w_start + CROP_SIZE]

    @staticmethod
    def _crop_image(args):
        image, position = args
        h, w, _ = image.shape
        if position == 'center':
            h_start = np.floor_divide(h - CROP_SIZE, 2)
            w_start = np.floor_divide(w - CROP_SIZE, 2)
        # TODO: crops
        if position == 'top-left':
            pass
        if position == 'top-right':
            pass
        if position == 'bottom-left':
            pass
        if position == 'bottom-right':
            pass

    @staticmethod
    def _pad_image(image):
        h, w, _ = image.shape
        delta_h = max(CROP_SIZE - h, 0)
        delta_w = max(CROP_SIZE - w, 0)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image

import cv2
import numpy as np
from skimage.exposure import adjust_gamma
from base_dataloader import DataLoader
from task_config import RESCALE_SIZE
from keras.applications import *

# known pretrained networks
CLF2MODULE = {
    'densenet40':   'densenet',
    'densenet121':  'densenet',
    'densenet169':  'densenet',
    'densenet201':  'densenet',
    'resnet50':     'resnet50',
    'xception':     'xception'
}
CLF2CLASS = {
    'densenet40':   'DenseNet40',
    'densenet121':  'DenseNet121',
    'densenet169':  'DenseNet169',
    'densenet201':  'DenseNet201',
    'resnet50':     'ResNet50',
    'xception':     'Xception'
}


class ImageLoader(DataLoader):

    def __init__(self, *args, **kwargs):
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
        image = cv2.resize(image, (RESCALE_SIZE, RESCALE_SIZE))
        return image

    def _augment_sample(self, image):
        # TODO: flip, rot90, gamma, color, zoom?
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
    def _crop_image(args):
        image, side_len, center = args
        h, w, _ = image.shape
        if h == side_len and w == side_len:
            return image
        assert h > side_len and w > side_len
        if center:
            h_start = np.floor_divide(h - side_len, 2)
            w_start = np.floor_divide(w - side_len, 2)
        else:
            h_start = np.random.randint(0, h - side_len)
            w_start = np.random.randint(0, w - side_len)
        return image[h_start:h_start + side_len, w_start:w_start + side_len]

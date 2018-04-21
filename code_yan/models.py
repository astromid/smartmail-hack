from keras.models import Model
from keras.layers import Dense, Dropout, Input
from image_loader import CLF2MODULE, CLF2CLASS
from task_config import RESCALE_SIZE
from keras.applications import *


class PretrainedCLF:

    def __init__(self, clf_name, n_class):
        self.clf_name = clf_name
        self.n_class = n_class
        self.module_ = CLF2MODULE[clf_name]
        self.class_ = CLF2CLASS[clf_name]
        self.backbone = getattr(globals()[self.module_], self.class_)

        i = self._input()
        print(f"Using {self.class_} as backbone")
        backbone = self.backbone(
            include_top=False,
            weights='imagenet',
            pooling='max'
        )
        x = backbone(i)
        out = self._top_classifier(x)
        self.model = Model(i, out)
        for layer in self.model.get_layer(self.clf_name).layers:
            layer.trainable = False

    @staticmethod
    def _input():
        input_ = Input((RESCALE_SIZE, RESCALE_SIZE, 3))
        return input_

    def _top_classifier(self, x):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(self.n_class, activation='sigmoid')(x)
        return out

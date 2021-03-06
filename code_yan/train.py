import os
import argparse
import numpy as np
import models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam
from glob import glob
from sklearn.model_selection import train_test_split
from image_loader import ImageLoader
from task_config import ROOT_DIR, TRAIN_DIR
from utils import LoggerCallback
from sklearn.metrics import accuracy_score, fbeta_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Name of the network in format {base_clf}-{number}")
    parser.add_argument('-e', '--epochs', type=int, help="Total # of epochs to train")
    parser.add_argument('-fe', '--frozen_epochs', type=int, help="# of epochs with frozen backbone")
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-val', '--validation_size', type=float, default=0.2, help="Percent of dataset for validation")
    parser.add_argument('-s', '--seed', type=int, default=12017952)
    parser.add_argument('-bal', '--balance', action='store_true', help="Enables sample class balancing")
    parser.add_argument('-aug', '--augmentation', action='store_true', help="Enables augmentation")
    parser.add_argument('-cr', '--crop', action='store_true', help="Use crops instead of brainless rescaling")
    args = parser.parse_args()

    np.random.seed(42)
    MODEL_DIR = os.path.join(ROOT_DIR, 'models', args.name)
    CLF_NAME = args.name.split('-')[0]
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_files = sorted(glob(os.path.join(TRAIN_DIR, '*', '*')))
    all_labels = [os.path.split(os.path.dirname(file))[-1] for file in all_files]
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42, stratify=all_labels)
    MODEL_PATH = os.path.join(MODEL_DIR, CLF_NAME)
    train_loader = ImageLoader(
        files=train_files,
        aug_config=args.augmentation,
        balance=args.balance,
        batch_size=args.batch_size,
        clf_name=CLF_NAME,
        crops=args.crop
    )
    val_loader = ImageLoader(
        files=val_files,
        mode='val',
        batch_size=args.batch_size,
        clf_name=CLF_NAME,
        crops=args.crop
    )
    assert train_loader.n_class == val_loader.n_class
    lr_cb = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=2e-5
    )
    log_cb = LoggerCallback(log_path=MODEL_PATH)
    model_compile_args = {
        'optimizer': Adam(),
        'loss': 'binary_crossentropy',
        'metrics': ['categorical_accuracy']
    }
    model = models.PretrainedCLF(CLF_NAME, train_loader.n_class).model
    model.compile(**model_compile_args)
    model.summary()
    frozen_history = model.fit_generator(
        generator=train_loader,
        steps_per_epoch=len(train_loader),
        epochs=args.frozen_epochs,
        verbose=1,
        callbacks=[lr_cb, log_cb],
        validation_data=val_loader,
        validation_steps=len(val_loader)
    )
    model.save(MODEL_PATH + f'-ep{args.frozen_epochs}.h5')
    print("Frozen model saved successfully")
    for layer in model.get_layer(CLF_NAME).layers:
        layer.trainable = True
    model.compile(**model_compile_args)
    model.summary()
    check_acc = ModelCheckpoint(
        filepath=MODEL_PATH + '-acc-best.h5',
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True
    )
    check_loss = ModelCheckpoint(
        filepath=MODEL_PATH + '-loss-best.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    lr_cb = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.3,
        patience=4,
        verbose=1,
        min_lr=1e-9
    )
    history = model.fit_generator(
        generator=train_loader,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        verbose=1,
        callbacks=[check_acc, check_loss, lr_cb, log_cb],
        validation_data=val_loader,
        validation_steps=len(val_loader),
        initial_epoch=args.frozen_epochs
    )
    model.save(MODEL_PATH + f"-ep{args.epochs}.h5")
    print("Models saved successfully")

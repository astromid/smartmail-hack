import os
import argparse
import numpy as np
from comet_ml import Experiment
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
from sklearn.metrics import accuracy_score, classification_report, fbeta_score


if __name__ == '__main__':
    experiment = Experiment("gaV3YiEaKRsBEXbkpMKaaRv8D")
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
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    train_files, val_files = train_test_split(train_val_files, test_size=0.1, random_state=42, stratify=train_val_labels)
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
    check_acc = ModelCheckpoint(
        filepath=MODEL_PATH + '-frozen-acc-best.h5',
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True
    )
    check_loss = ModelCheckpoint(
        filepath=MODEL_PATH + '-frozen-loss-best.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    lr_cb = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-5
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
    with experiment.train():
        frozen_history = model.fit_generator(
            generator=train_loader,
            steps_per_epoch=len(train_loader),
            epochs=args.frozen_epochs,
            verbose=1,
            callbacks=[check_acc, check_loss, lr_cb, log_cb],
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
            factor=0.5,
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
    test_loader = ImageLoader(
        files=test_files,
        mode='test',
        batch_size=args.batch_size,
        clf_name=CLF_NAME,
        crops=args.crop
    )
    model_files = sorted(glob(os.path.join(MODEL_DIR, '*.h5')))
    print(f"Found {len(model_files)} models to evaluate")
    for file in model_files:
        K.clear_session()
        model_name = os.path.basename(file)
        model = load_model(file)
        print("-" * 25)
        print(f"Model: {model_name}")
        with experiment.test():
            probs = model.predict_generator(
                generator=test_loader,
                steps=len(test_loader),
                verbose=1)
            ids = np.argmax(probs, axis=1)
            labels = [test_loader.id2label[id_] for id_ in ids]
            accuracy = accuracy_score(test_loader.labels, labels)
            print(f"accuracy: {accuracy}")
            experiment.log_metric(model_name + '_accuracy', accuracy)
            f2 = fbeta_score(test_loader.labels, labels, 2.0, average='macro')
            print(f"F2: {f2}")
            experiment.log_metric(model_name + '_F2', f2)
            print(classification_report(test_loader.labels, labels))

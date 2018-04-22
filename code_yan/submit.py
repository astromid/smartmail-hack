import os
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
from glob import glob
from image_loader import ImageLoader
from task_config import ROOT_DIR, TEST_DIR, TASK_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Name of the network in format {base_clf}-{number}")
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    MODEL_DIR = os.path.join('models', args.name + '-' + TASK_NAME)
    SUB_DIR = os.path.join(ROOT_DIR, 'subs')
    TEST_CONFIG = {
        'batch_size': args.batch_size,
        'clf_name': args.name.split('-')[0]
    }
    test_files = sorted(glob(os.path.join(TEST_DIR, '*', '*')))
    test_loader = ImageLoader(files=test_files, mode='test', **TEST_CONFIG)
    model_files = sorted(glob(os.path.join(MODEL_DIR, '*.h5')))
    print(f"Found {len(model_files)} models to evaluate")
    for file in model_files:
        K.clear_session()
        model_name = os.path.basename(file)
        model = load_model(file)
        print(f"Model: {model_name}")
        probs = model.predict_generator(
            generator=test_loader,
            steps=len(test_loader),
            verbose=1)
        ids = np.argmax(probs, axis=1)
        labels = [test_loader.id2label[id_] for id_ in ids]
        data = {
            'file': test_loader.files,
            'label': labels
        }
        sub_path = os.path.join(SUB_DIR, f'{file}-sub.csv')
        pd.DataFrame(data).to_csv(sub_path, index=False)
        print(f"Submit {sub_path} created successfully")

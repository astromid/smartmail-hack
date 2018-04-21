# TODO: this file need to be rewrited for smartmail hack task
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import load_model
from keras import backend as K
from glob import glob
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from image_loader import ImageLoader
from task_config import TEST_DIR, TASK_NAME

plt.switch_backend('agg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Name of the network in format {base_clf}-{number}")
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    MODEL_DIR = os.path.join('models', args.name + '-' + TASK_NAME)
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
        print("-" * 25)
        print(f"Model: {model_name}")
        probs = model.predict_generator(
            generator=test_loader,
            steps=len(test_loader),
            verbose=1)
        ids = np.argmax(probs, axis=1)
        labels = [test_loader.id2label[id_] for id_ in ids]
        accuracy = accuracy_score(test_loader.labels, labels)
        print(f"accuracy: {accuracy}")
        print(classification_report(test_loader.labels, labels))
        if test_loader.n_class == 2:
            y_true = [test_loader.label2id[label] for label in test_loader.labels]
            roc_auc = roc_auc_score(y_true, probs[:, 1])
            print(f"ROC-AUC: {roc_auc}")

        pp = PdfPages(f'{file}.pdf')
        mis_idx = [idx for idx in range(len(labels)) if test_loader.labels[idx] != labels[idx]]
        mis_probs_max = np.max(probs, axis=1)[mis_idx]
        mis_probs = probs[mis_idx]
        mis_real_labels = np.array(test_loader.labels)[mis_idx]
        mis_pred_labels = np.array(labels)[mis_idx]
        mis_images = np.array(test_loader.samples)[mis_idx]
        top_mistakes = mis_probs_max.argsort()[-10:][::-1]
        for idx in top_mistakes:
            fig = plt.figure(figsize=(16, 10))
            plt.title(f"Real: {mis_real_labels[idx]}, pred: {mis_pred_labels[idx]}")
            plt.subplot(1, 2, 1)
            plt.imshow(mis_images[idx])
            plt.subplot(1, 2, 2)
            sns.barplot(x=test_loader.possible_labels, y=mis_probs[idx])
            pp.savefig()
            plt.close(fig)
        pp.close()

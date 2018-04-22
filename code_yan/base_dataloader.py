import os
import numpy as np
from abc import abstractmethod
from keras.utils import Sequence, to_categorical
from multiprocessing.pool import ThreadPool
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

# supported modes
LOAD_MODES = ['full', 'only_batch']
DATA_MODES = ['train', 'val', 'test']


class DataLoader(Sequence):

    def __init__(self, files, balance=False,
                 batch_size=16, aug_config=False,
                 clf_name=None, load_mode='full',
                 mode='train'):
        self.files = sorted(files)
        self.balance = balance
        self.batch_size = batch_size
        self.aug_config = aug_config
        self.clf_name = clf_name
        self.load_mode = load_mode
        self.mode = mode
        self.p = ThreadPool()
        self._preprocess_batch = None

        if self.load_mode not in LOAD_MODES:
            print(f"{self.load_mode} load mode is not supported yet")
            print(f"Supported load modes: {LOAD_MODES}")
            raise NameError

        if self.mode not in DATA_MODES:
            print(f"{self.mode} mode is not correct")
            print(f"Correct modes: {DATA_MODES}")
            raise NameError

        self.samples = self.files
        self.labels = [os.path.split(os.path.dirname(file))[-1] for file in self.samples]
        self.len_ = len(self.samples)
        self.possible_labels = sorted(np.unique(self.labels))
        self.n_class = len(self.possible_labels)
        print(f"Found {self.n_class} labels")
        self.id2label = {i: label for i, label in enumerate(self.possible_labels)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        # load all files in memory
        if self.load_mode == 'full':
            loaded_samples = []
            with tqdm(desc=f"Loading {self.mode} files", total=self.len_) as pbar:
                for sample in self.p.imap(self._load_sample, self.samples):
                    loaded_samples.append(sample)
                    pbar.update()
            assert len(loaded_samples) == len(self.labels)
            # clean-up failed samples (None)
            print("Clean-up failed samples:")
            for idx in range(self.len_):
                if loaded_samples[idx] is None:
                    loaded_samples.pop(idx)
                    dropped_label = self.labels.pop(idx)
                    print(f"Label {dropped_label} for file {self.samples[idx]} has been dropped")
            self.samples = loaded_samples
            self.len_ = len(self.samples)
            print(f"Successfully loaded {self.len_} {self.mode} samples")
        # initial shuffle for train mode
        self.on_epoch_end()

    def __len__(self):
        return np.ceil(self.len_ / self.batch_size).astype('int')

    def __getitem__(self, idx):
        x = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.load_mode == 'only_batch':
            x = [sample for sample in self.p.imap(self._load_sample, x)]
        x = [sample for sample in self.p.imap(self._prepare_sample, x)]
        if self.aug_config:
            x = [sample for sample in self.p.imap(self._augment_sample, x)]
        x = np.array(x).astype(np.float32)
        if self._preprocess_batch is not None:
            x = self._preprocess_batch(x)
        if self.mode != 'test':
            y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
            label_ids = [self.label2id[label] for label in y]
            y = to_categorical(label_ids, self.n_class)
            if self.balance:
                weights = compute_sample_weight('balanced', label_ids)
                return x, y, weights
            else:
                return x, y
        else:
            return x

    def on_epoch_end(self):
        if self.mode == 'train':
            data = list(zip(self.samples, self.labels))
            np.random.shuffle(data)
            self.samples, self.labels = zip(*data)
            self.samples = list(self.samples)
            self.labels = list(self.labels)

    @abstractmethod
    def _load_sample(self, sample_path):
        raise NotImplementedError

    @abstractmethod
    def _prepare_sample(self, sample):
        raise NotImplementedError

    @abstractmethod
    def _augment_sample(self, sample):
        raise NotImplementedError

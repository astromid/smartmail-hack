import numpy as np
from datetime import datetime
from keras.callbacks import Callback


# simple keras callback for logging in file
class LoggerCallback(Callback):

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        with open(self.log_path + '-train.log', 'a') as log_file:
            log_file.write("-------------------------------\n")
            log_file.write(f"Train started at {datetime.now()}\n")

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.params['metrics']
        metric_format = "{name}: {value:0.5f}"
        strings = [metric_format.format(
            name=metric,
            value=np.mean(logs[metric], axis=None)
        ) for metric in metrics if metric in logs]
        epoch_output = "Epoch {ep:05d}: ".format(ep=(epoch + 1))
        output = epoch_output + ', '.join(strings)
        with open(self.log_path + '-train.log', 'a') as log_file:
            log_file.write(output + '\n')

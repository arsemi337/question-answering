from timeit import default_timer as timer
import tensorflow as tf


class TimeMeasureCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_train_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        total_time = timer() - self.start_time

        self.epoch_train_times.append(total_time)

    def total_training_time(self):
        return sum(self.epoch_train_times)

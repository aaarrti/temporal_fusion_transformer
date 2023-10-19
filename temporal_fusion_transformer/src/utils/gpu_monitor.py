from absl import logging as log
from absl_extra.cuda_utils import get_memory_info
from keras.utils import TimedThread
from keras_core.callbacks import Callback


class MonitorThread(TimedThread):
    def on_interval(self):
        memory_info = get_memory_info()
        log.info(f"{memory_info = }")


class MonitorGPUMemory(Callback):
    def __init__(self, interval=30):
        super().__init__()
        self.thread = MonitorThread(interval=interval)

    def on_train_begin(self, logs=None):
        self.thread.start()

    def on_train_end(self, logs=None):
        self.thread.stop()

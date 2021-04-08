#from keras import backend as K
#from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
import pickle

""" -----------------------------------------------------------------------------------------
Callback that records events into a `History` object. It then saves the history after each 
epoch into a file.
To read the file into a python dict:
            history = {}
            with open(filename, "r") as f:
                history = eval(f.read())
This may be unsafe since eval() will evaluate any string; A safer alternative:
        import ast
        history = {}
        with open(filename, "r") as f:
            history = ast.literal_eval(f.read())
https://github.com/titu1994/Image-Super-Resolution
----------------------------------------------------------------------------------------- """ 
class HistoryCheckpoint(Callback):

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "wb") as f:
            pickle.dump(self.history, f)
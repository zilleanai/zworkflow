
class PredictBase(object):
    def __init__(self, config):
        self.config = config

    def predict(self, data, model, verbose=False):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class EvaluateBase(object):
    def __init__(self, config):
        self.config = config

    def evaluate(self, dataset, model, verbose=False):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

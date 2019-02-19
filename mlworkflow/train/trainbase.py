
class TrainBase(object):
    def __init__(self, config):
        self.config = config

    def train(self, dataset, model):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

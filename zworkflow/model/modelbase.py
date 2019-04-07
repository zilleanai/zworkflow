
class ModelBase(object):
    def __init__(self, config):
        self.config = config

    def __str__(self):
        raise NotImplementedError

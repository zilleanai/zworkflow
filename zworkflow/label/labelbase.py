
class LabelBase(object):
    def __init__(self, config):
        self.config = config

    def label(self, verbose=False):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

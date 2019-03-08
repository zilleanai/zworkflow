
class PreprocessingBase():
    def __init__(self, config):
        self.config = config

    def process(self, data, verbose=False):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

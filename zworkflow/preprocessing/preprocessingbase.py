
class PreprocessingBase():

    functions = {}

    def __init__(self, config):
        self.config = config

    def process(self, data, verbose=False):
        raise NotImplementedError

    def keys(self):
        return self.functions.keys()

    def __str__(self):
        raise NotImplementedError

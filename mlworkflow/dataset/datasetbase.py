
class DataSetBase(object):
    def __init__(self, config):
        self.config = config

    def __getitem__(self, idx):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

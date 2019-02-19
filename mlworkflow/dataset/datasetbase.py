
class DataSetBase():
    def __init__(self, config):
        self.config = config
        self.current = 0

    def __getitem__(self, idx):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        if self.current + 1 >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self[self.current]

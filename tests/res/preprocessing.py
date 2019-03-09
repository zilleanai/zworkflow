
from mlworkflow.preprocessing import PreprocessingBase

class preprocessing(PreprocessingBase):
    def __init__(self, config):
        super().__init__(config)
        self.functions['move'] = self.move

    def move(self, data):
        data['move'] = data['price'] - data['price'].shift(10)
        return data

    def process(self, data, verbose=False):
        for f in self.config['preprocessing']['functions']:
            fun = self.functions[f]
            data = fun(data)
        return data

    def __str__(self):
        return "preprocessing: " + self.functions.keys()

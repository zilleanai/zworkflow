class preprocessing():
    def __init__(self, config):
        self.config = config

    def process(self, data, verbose=False):
        data['move'] = data['price'] - data['price'].shift(10)
        return data

    def __str__(self):
        return "preprocessing: ['move']"

import yaml


class Config():

    keys = []
    current = -1

    default = {
        'dataset': {
            'datapath': '.',
            'type': 'csv'
        },
        'model': {
            'dim_size': 1
        },
        'train': {
            'epochs': 2,
            'learn_rate': 0.01
        }
    }

    def __init__(self, config={}):
        """
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as file:
                self.config = yaml.load(file)
                self.filename = config

        self.fill_missing(self.config)
        self.keys = list(self.config.keys())

    def __setitem__(self, key, item):
        self.config[key] = item

    def __getitem__(self, key):
        return self.config[key]

    def __len__(self):
        return len(self.config)

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        if self.current + 1 >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self.keys[self.current]

    def __str__(self):
        return str(self.config)

    def fill_missing(self, config):
        for key in self.default.keys():
            if config.get(key) is None:
                config[key] = self.default[key]

    def save(self, path):
        """
        save current config to given path as yaml file
        """
        with open(path, 'w') as outfile:
            yaml.dump(self.config,
                      outfile, default_flow_style=False)

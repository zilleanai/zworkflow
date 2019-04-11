import yaml
from .configs import get_config

class Config():

    keys = []
    current = -1

    def __init__(self, config={}):
        """
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
                self.filename = config
        self.default = get_config(self.config.get('domain'))

        self.fill_missing(self.config, 'general')
        self.fill_missing(self.config, 'dataset')
        self.fill_missing(self.config, 'preprocessing')
        self.fill_missing(self.config, 'model')
        self.fill_missing(self.config, 'train')
        self.fill_missing(self.config, 'predict')
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

    def fill_missing(self, config, group):
        if config.get(group) is None:
            config[group] = {}
        for key in self.default[group].keys():
            if config[group].get(key) is None:
                config[group][key] = self.default[group][key]

    def save(self, path):
        """
        save current config to given path as yaml file
        """
        with open(path, 'w') as outfile:
            yaml.dump(self.config,
                      outfile, default_flow_style=False)

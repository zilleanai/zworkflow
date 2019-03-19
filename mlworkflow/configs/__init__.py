from .default import default

configs = {
    'default': default
}


def get_config(domain='default'):
    return configs.get(domain) or default

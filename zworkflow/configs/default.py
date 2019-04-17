
default = {
    'domain': 'default',
    'general': {
        'verbose': False
    },
    'dataset': {
        'dataset_class': 'dataset',
        'dataset_file': 'dataset.py',
        'datapath': '.',
        'type': 'csv'
    },
    'preprocessing': {
        'preprocessing_class': 'preprocessing',
        'preprocessing_file': 'preprocessing.py',
        'functions': []
    },
    'model': {
        'model_class': 'model',
        'model_file': 'model.py',
        'savepath': 'model.save',
        'dim_size': 1
    },
    'train': {
        'train_class': 'train',
        'train_file': 'train.py',
        'epochs': 2,
        'learn_rate': 0.01,
        'batch_size': 10,
        'load_model': True,
        'save_every_epoch': 1,
        'device': 'cpu'
    },
    'predict': {
        'predict_class': 'predict',
        'predict_file': 'predict.py'
    },
    'label': {
        'label_class': 'label',
        'label_file': 'label.py'
    },
    'evaluate': {
        'evaluate_class': 'evaluate',
        'evaluate_file': 'evaluate.py'
    }
}

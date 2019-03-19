bayesian_optimization = {
    'domain': 'bayesian_optimization',
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
        'model_class': 'bayesian_optimization_model',
        'model_file': 'model.py',
        'savepath': 'bayesian_optimization_model.save',
        'dim_size': 1
    },
    'train': {
        'train_class': 'bayesian_optimization_train',
        'train_file': 'train.py',
        'epochs': 2,
        'learn_rate': 0.01,
        'batch_size': 10,
        'load_model': True,
        'save_every_epoch': 1,
        'device': 'cpu'
    },
    'predict': {
        'predict_class': 'bayesian_optimization_predict',
        'predict_file': 'predict.py'
    }
}

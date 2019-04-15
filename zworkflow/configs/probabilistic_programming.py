probabilistic_programming = {
    'domain': 'probabilistic_programming',
    'general': {
        'verbose': False
    },
    'dataset': {
        'dataset_class': 'dataset',
        'dataset_file': 'dataset.py',
        'datapath': '.',
        'features':[],
        'labels':[],
        'type': 'csv'
    },
    'preprocessing': {
        'preprocessing_class': 'preprocessing',
        'preprocessing_file': 'preprocessing.py',
        'functions': []
    },
    'model': {
        'model_class': 'probabilistic_programming_model',
        'model_file': 'probabilistic_programming_model.py',
        'savepath': 'probabilistic_programming_model.save',
        'dim_size': 1
    },
    'train': {
        'train_class': 'probabilistic_programming_train',
        'train_file': 'probabilistic_programming_train.py',
        'epochs': 2,
        'learn_rate': 0.01,
        'batch_size': 10,
        'load_model': True,
        'save_every_epoch': 1,
        'device': 'cpu'
    },
    'predict': {
        'predict_class': 'probabilistic_programming_predict',
        'predict_file': 'probabilistic_programming_predict.py'
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

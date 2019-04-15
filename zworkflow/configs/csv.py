
csv = {
    'domain': 'csv',
    'general': {
        'verbose': False
    },
    'dataset': {
        'dataset_class': 'csv_dataset',
        'dataset_file': 'dataset.py',
        'datapath': '.',
        'type': 'csv'
    },
    'preprocessing': {
        'preprocessing_class': 'csv_preprocessing',
        'preprocessing_file': 'preprocessing.py',
        'functions': []
    },
    'model': {
        'model_class': 'csv_model',
        'model_file': 'model.py',
        'savepath': 'pp_model.save',
        'dim_size': 1
    },
    'train': {
        'train_class': 'csv_train',
        'train_file': 'train.py',
        'epochs': 10,
        'learn_rate': 0.001,
        'batch_size': 100,
        'load_model': True,
        'save_every_epoch': 2,
        'device': 'cpu'
    },
    'predict': {
        'predict_class': 'csv_predict',
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

segmentation = {
    'domain': 'segmentation',
    'general': {
        'verbose': False
    },
    'dataset': {
        'dataset_class': 'segmentation_dataset',
        'dataset_file': 'dataset.py',
        'train_images': 'train',
        'train_masks': 'train_masks',
        'width': 224,
        'height': 224,
        'outwidth': 224,
        'outheight': 224,
        'classes': 1
    },
    'preprocessing': {
        'preprocessing_class': 'segmentation_preprocessing',
        'preprocessing_file': 'preprocessing.py',
        'functions': ['resize', 'normalize', 'classify_mask']
    },
    'model': {
        'model_class': 'segmentation_model',
        'model_file': 'model.py',
        'savepath': 'segmentation_model.save',
        'dim_size': 3
    },
    'train': {
        'train_class': 'segmentation_train',
        'train_file': 'train.py',
        'epochs': 2,
        'learn_rate': 0.01,
        'batch_size': 4,
        'load_model': True,
        'save_every_epoch': 1,
        'device': 'cpu',
        'shuffle': True
    },
    'predict': {
        'predict_class': 'segmentation_predict',
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

from GlobalValues import  BATCH_SIZE, EPOCHS, LEARNING_RATE

config1 = {
        LEARNING_RATE: 5e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16
    }

# linear inter
config_goodResult = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16
    }

sweep_configuration1 = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters':
            {
                BATCH_SIZE: {'values': [8, 16]},
                EPOCHS: {'values': [100, 150, 200, 250]},
                LEARNING_RATE: {'max': 0.0001, 'min': 0.000001}
            }
    }

sweep_configuration_beta = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters':
            {
                BATCH_SIZE: {'values': [ 16]},
                EPOCHS: {'values': [150]},
                LEARNING_RATE: {'values': [0.00003]},
                'beta': {'max': 1, 'min': 0}
            }
    }


# cubic inter
# config = {
#     'learning_rate': 6e-7,
#     'epochs': 150,
#     'batch_size': 16
# }
# config = {
#     'learning_rate': 6e-7,
#     'epochs': 100,
#     'batch_size': 16
# }
# config = {
#         'learning_rate': 3e-7,
#         'epochs': 150,
#         'batch_size': 8
#     }
# config = {
#     'learning_rate': 3e-7,
#     'epochs': 150,
#     'batch_size': 8
# }


# 8 batch size works best for 4000 records or less
# 3e-5
config_IOU = {
        'learning_rate': 3e-7,
        'epochs': 150,
        'batch_size': 8
    }

config_IOU_hl = {
        'learning_rate': 5e-6,
        'epochs': 150,
        'batch_size': 8
    }
config_IOU_vhl = {
        'learning_rate': 5e-5,
        'epochs': 150,
        'batch_size': 8
    }
sweep_configuration_IOU = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'batch_size': {'values': [8,16]},
        'epochs': {'values': [100,150,200,250]},
        'learning_rate': {'max': 0.00001, 'min': 0.000001}
     }
}
"""
Hyperparametrs and sweep

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""
from GlobalValues import BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA, LOSS_FUNCTION
from LossFunctions import MSEiou2, IOU_number, IOU_nonBinary, MSEunion, MSEintersection, IMSE2, LMSE, GMSE, MSENew, \
    jacard_loss

config1 = {
        LEARNING_RATE: 5e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEiou2
    }

config_IMSE2 = {
        LEARNING_RATE: 5e-6,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : IMSE2
    }

config_GMSE= {
        LEARNING_RATE: 3e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : GMSE
    }

config_LMSE= {
        LEARNING_RATE: 3e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : LMSE
    }
# linear inter
config_goodResult = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEiou2
    }

config_goodResult2 = {
        LEARNING_RATE: 3e-6,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEiou2
    }

config_MSENew = {
        LEARNING_RATE: 1e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSENew
    }
config_MSEintersection = {
        LEARNING_RATE: 1e-7,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEintersection
    }
config_MSEunion = {
        LEARNING_RATE: 1e-7,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEunion
    }
config_MSEunion2 = {
        LEARNING_RATE: 7e-6,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : MSEiou2
    }

config_IOU_number = {
        LEARNING_RATE: 9e-6,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : IOU_number
    }

config_IOU_nonBinary = {
        LEARNING_RATE: 3e-6,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : IOU_nonBinary
    }


config_IOUMNS_goodResult = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 100,
        BATCH_SIZE: 16,
        BETA: 0.85,
        LOSS_FUNCTION : MSEiou2
    }
config_IOUMNS_goodResult2 = {
        LEARNING_RATE: 2e-5,
        EPOCHS: 1,
        BATCH_SIZE: 16,
        BETA: 0.5

    }

config_jacard_loss = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 200,
        BATCH_SIZE: 16,
        LOSS_FUNCTION : jacard_loss

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
                BETA: {'max': 1, 'min': 0}
            }
    }

# 8 batch size works best for 4000 records or less
# 3e-5
config_IOU = {
        'learning_rate': 5e-6,
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

sweep_configuration_IOU_LRMSE = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        BATCH_SIZE: {'values': [8,16]},
        EPOCHS: {'values': [100,150,200,250]},
        LEARNING_RATE: {'max': 0.00005, 'min': 0.0000001},
        BETA: {'max': 1.0, 'min': 0.4}
     }
}
use_config = config_jacard_loss

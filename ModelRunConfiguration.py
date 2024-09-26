"""
Hyperparametrs and sweep

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

from GlobalValues import BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA, LOSS_FUNCTION
from LossFunctions import GMSE, LRMSE, Classification_loss, Segmentation_loss, jaccard_loss, two_branch_loss, GLMSE
from Unet import UNET

model_list = [UNET]
Selected_model = model_list[0]

selected_case = 7

def get_HyperParams(selected_case):
    loss_cases = [
    # case 1 : GMSE
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GMSE
    },
    # case 2 : GLMSE global and local rmse
    {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GLMSE,
        BETA: 0.1
        #     beta = W_local_rmse / (w_local_rmse + w_global_rmse)
    },
    # case 3: jaccard_loss
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: jaccard_loss
    },
    # case 4: two_branch_loss
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: two_branch_loss,
        BETA: 0.81
        #     beta = W_rmse / (w_rmse + w_jaccard)
    },
    # case 5: local
     {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: LRMSE
    }
    ,
    # case 6: Classification_loss
     {
        LEARNING_RATE: 5e-3,
        EPOCHS: 150,
        BATCH_SIZE: 64,
        LOSS_FUNCTION: Classification_loss
    },
    # {
    #     LEARNING_RATE: 3e-5,
    #     EPOCHS: 150,
    #     BATCH_SIZE: 32,
    #     LOSS_FUNCTION: Classification_loss
    # },
    # case 7: Segmentation_loss
     {
        LEARNING_RATE: 8e-5,
        EPOCHS: 150,
        BATCH_SIZE: 32,
        LOSS_FUNCTION: Segmentation_loss
    },
    # case 8 : GMSE small
     {
        LEARNING_RATE: 3e-6,
        EPOCHS: 150,
        BATCH_SIZE: 64,
        LOSS_FUNCTION: GMSE
    },
     # case 9 : GMSE after seg
     {
        LEARNING_RATE: 1e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GMSE
    },
    # case 10: Segmentation_loss
     {
        LEARNING_RATE: 1e-6,
        EPOCHS: 200,
        BATCH_SIZE: 32,
        LOSS_FUNCTION: Segmentation_loss
    },
    # case 11: Segmentation_loss
     {
        LEARNING_RATE: 9e-5,
        EPOCHS: 150,
        BATCH_SIZE: 32,
        LOSS_FUNCTION: Segmentation_loss
    }
]
    
    return loss_cases[selected_case - 1]


use_config = get_HyperParams(selected_case)
use_config_UNET = get_HyperParams(7)


real_time_config = {
        LEARNING_RATE: 3e-5,
        EPOCHS: 150,
        BATCH_SIZE: 16,
        LOSS_FUNCTION: GMSE
    }

# ----------------------------------------------------------------------------------------------
# run multiple runs
SWEEP_OPERATION = False
sweep_loss_funtion = GMSE
sweep_configuration_IOU_LRMSE = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
        {
            BATCH_SIZE: {'values': [16,32]},
            EPOCHS: {'values': [150]},
            LEARNING_RATE: {'max': 0.00009, 'min': 0.00001},
            BETA: {'values': [0.8]}
        }
}

# sweep_configuration_IOU_LRMSE = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric': {'goal': 'minimize', 'name': 'val_loss'},
#     'parameters':
#         {
#             BATCH_SIZE: {'values': [16]},
#             EPOCHS: {'values': [150]},
#             LEARNING_RATE: {'max': 0.00005, 'min': 0.0000001},
#             BETA: {'values': [150]}
#         }
# }



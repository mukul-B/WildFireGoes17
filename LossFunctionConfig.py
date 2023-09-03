"""
Hyperparametrs and sweep

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

from GlobalValues import BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA, LOSS_FUNCTION
from LossFunctions import GMSE, LRMSE, jaccard_loss, two_branch_loss, GLMSE

selected_case = 4
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
]
use_config = loss_cases[selected_case - 1]

# ----------------------------------------------------------------------------------------------
# run multiple runs
SWEEP_OPERATION = False
sweep_loss_funtion = two_branch_loss
sweep_configuration_IOU_LRMSE = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
        {
            BATCH_SIZE: {'values': [16]},
            EPOCHS: {'values': [150]},
            LEARNING_RATE: {'max': 0.0003, 'min': 0.00003},
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



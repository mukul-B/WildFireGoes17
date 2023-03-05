"""
Hyperparametrs and sweep

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

from GlobalValues import BATCH_SIZE, EPOCHS, LEARNING_RATE, BETA, LOSS_FUNCTION
from LossFunctions import GMSE, jaccard_loss, two_branch_loss, GLMSE

selected_case = 4
# case 1 : GMSE
config_GMSE = {
    LEARNING_RATE: 3e-5,
    EPOCHS: 150,
    BATCH_SIZE: 16,
    LOSS_FUNCTION: GMSE
}

# case 2 : GLMSE global and local rmse
config_GLMSE = {
    LEARNING_RATE: 3e-5,
    EPOCHS: 150,
    BATCH_SIZE: 16,
    LOSS_FUNCTION: GLMSE,
    BETA: 0.1
    #     beta = W_local_rmse / (w_local_rmse + w_global_rmse)
}

# case 3: jaccard_loss
config_JACCARD = {
    LEARNING_RATE: 3e-5,
    EPOCHS: 150,
    BATCH_SIZE: 16,
    LOSS_FUNCTION: jaccard_loss

}
# case 4: two_branch_loss
config_TWO_BRANCH = {
    LEARNING_RATE: 3e-5,
    EPOCHS: 150,
    BATCH_SIZE: 16,
    LOSS_FUNCTION: two_branch_loss,
    BETA: 0.9
    #     beta = W_rmse / (w_rmse + w_jaccard)
}

# run multiple runs
sweep_configuration_IOU_LRMSE = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
        {
            BATCH_SIZE: {'values': [8, 16]},
            EPOCHS: {'values': [100, 150, 200, 250]},
            LEARNING_RATE: {'max': 0.00005, 'min': 0.0000001},
            BETA: {'max': 1.0, 'min': 0.4}
        }
}

loss_cases = [config_GMSE,
              config_GLMSE,
              config_JACCARD,
              config_TWO_BRANCH
              ]
use_config = loss_cases[selected_case - 1]

"""
This Config file contains all the global constant used in Project

Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""
#  constants
FDC = "ABI-L2-FDCC"
RAD = "ABI-L1b-RadC"
CMI = "ABI-L2-CMIPC"
BETA = 'beta'
LC = 'LC'
HC = 'HC'
LI = 'LI'
HI = 'HI'
GOES_MIN_VAL, GOES_MAX_VAL = 210 , 413
VIIRS_MIN_VAL,VIIRS_MAX_VAL = 0 , 367
VIIRS_UNITS ='Brightness Temperature'
# GOES_UNITS = 'Radiance'
GOES_UNITS = 'Brightness Temperature'
PREDICTION_UNITS = 'Brightness Temperature'
RES_OPT_PTH = 'SuperRes_Opt.pth'
RES_DECODER_PTH = 'SuperRes_Decoder.pth'
RES_ENCODER_PTH = 'SuperRes_Encoder.pth'
LEARNING_RATE = 'learning_rate'
BATCH_SIZE = 'batch_size'
LOSS_FUNCTION = 'loss_function'
EPOCHS = 'epochs'
random_state = 42

# files and directories
GOES_ndf = 'GOES_netcdfs'
goes_folder = "GOES"
viirs_folder = "VIIRS"
logs = 'logs'
data_dir = "reference_data"
compare = 'compare'

# data loading and preprocessing
# site_conf = 'config/configuration_2019.yml'
# toExecuteSiteList = "config/training_sites_2019"
site_conf = 'config/conf_sites.yml'
toExecuteSiteList = "config/training_sites"
# toExecuteSiteList = "config/testing_sites"
# training = 'training'
# reference_data = "reference_data_working"
reference_data = "reference_data"
compare_dir = f'{reference_data}/$LOC/compare/'
viirs_dir = f'{reference_data}/$LOC/VIIRS/'
goes_dir = f'{reference_data}/$LOC/GOES/$PROD/$BAND/tif/'
# training_dir = 'training_data_workingwithFRP/'
training_dir = 'training_data/'
# training_dir = 'training_data_working/'

# Autoencoder training and testing
# model_path = 'Model_BEFORE_MOVING_NORMALIZATION/'
model_path = 'Model/'
Results = 'results/'
# THRESHOLD_COVERAGE = 0.2
# THRESHOLD_IOU = 0.05
THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.453186035,0.005117899


# toExecuteSiteList = "config/testing_sites"
testing_dir = 'testing_dir/'
# realtimeSiteList = "config/realtime_sites"
RealTimeIncoming_files = 'RealTimeIncoming_files/'
RealTimeIncoming_results = 'RealTimeIncoming_results/'
videos = 'Videos/'

# blind testing
realtimeSiteList = "config/blind_testing_sites"






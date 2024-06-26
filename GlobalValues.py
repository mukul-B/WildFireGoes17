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

# GOES_product = RAD
GOES_product = [{'product_name': RAD, 'band': 7},
           {'product_name': RAD, 'band': 14},
           {'product_name': RAD, 'band': 15}]
# GOES_product = [{'product_name': RAD, 'band': 7}]
GOES_product_size = len(GOES_product)
GOES_Bands = 2
gf_c_fields = [f'gf_c{i+1}' for i in range(GOES_Bands)]
training_data_field_names = ['vf'] + gf_c_fields + ['vf_FRP', 'gf_min', 'gf_max', 'vf_max']

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
data_dir = "DataRepository/reference_data"
compare = 'compare'

# data loading and preprocessing
# site_conf = 'config/configuration_2019.yml'
# toExecuteSiteList = "config/training_sites_2019"
site_conf = 'config/conf_sites.yml'
toExecuteSiteList = "config/training_sites"
# toExecuteSiteList = "config/testing_sites"
# training = 'training'
# reference_data = "reference_data_working"
reference_data = "DataRepository/reference_data"
compare_dir = f'{reference_data}/$LOC/compare/'
viirs_dir = f'{reference_data}/$LOC/VIIRS/'
goes_dir = f'{reference_data}/$LOC/GOES/$PROD_BAND/tif/'
# training_dir = 'training_data_workingwithFRP/'
training_dir = 'DataRepository/training_data/'
# training_dir = 'training_data_working/'

# Autoencoder training and testing
# model_path = 'Model_BEFORE_MOVING_NORMALIZATION/'
model_path = 'Model/'
Results = 'DataRepository/results/'
# THRESHOLD_COVERAGE = 0.2
# THRESHOLD_IOU = 0.05
THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.453186035,0.005117899


# toExecuteSiteList = "config/testing_sites"
testing_dir = 'DataRepository/testing_dir/'
# realtimeSiteList = "config/realtime_sites"
RealTimeIncoming_files = 'DataRepository/RealTimeIncoming_files/'
RealTimeIncoming_results = 'DataRepository/RealTimeIncoming_results/'
videos = 'Videos/'

# blind testing
realtimeSiteList = "config/blind_testing_sites"

paper_results = ['713','122','956','728','118','553','408','387','849','104','663','609']
NO_SAMPLES = []
ALL_SAMPLES = 0
SELECTED_SAMPLES = paper_results


# if filename[0] in ['79', '126', '199', '729', '183', '992', '140', '189', '1159', '190', '26', '188']:
# if filename[0] in ['78','240','249','0','6','19','2','10','14','15','27','807']:
# if filename[0] in ['401','237','122','713','792','821','888','358','728','626','943','594','969','118','395','730','444','408','387','204','296','774','93','882','720','823','280','859','809','115','952','849','956','884','156','171','104','663','396']:
# if filename[0] in ['713','122','956','728','118','553','408','387','849','104','663','609']:
# if filename[0] in ['24']







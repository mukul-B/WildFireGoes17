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
COLOR_NORMAL_VALUE = 1

# GOES_MIN_VAL, GOES_MAX_VAL = [210, 207, 205],[413,342, 342]
GOES_MIN_VAL, GOES_MAX_VAL = [209, 202, 198],[413,342, 342]
# GOES_MIN_VAL, GOES_MAX_VAL = 210 , 413
VIIRS_MIN_VAL,VIIRS_MAX_VAL = 0 , 367
VIIRS_UNITS ='Brightness Temperature'

# GOES_product = RAD
# GOES_product = [{'product_name': RAD, 'band': 1},
#                 {'product_name': RAD, 'band': 2},
#                 {'product_name': RAD, 'band': 3},
#                 {'product_name': RAD, 'band': 7},
#                  {'product_name': RAD, 'band': 14}]
GOES_product = [{'product_name': RAD, 'band': 7},
           {'product_name': RAD, 'band': 14},
           {'product_name': RAD, 'band': 15}]
# GOES_product = [{'product_name': RAD, 'band': 7}]
GOES_product_size = len(GOES_product)
GOES_Bands = 3
seperate_th,th_neg = 1,0

no_postfix = ''
_3channel_postfix = '_3channel'
_1channel_postfix = '_1channel'
_2channel_postfix = '_activeFire_2branch'
_1channel_norm_postfix = '_1channel_perimageNormalized'
_3channel_classifier_postfix = '_3channel_classifier'
_3channel_precision_postfix = '_3channel_precision'
_3channel_precision_ESPG_postfix = '_3channel_precision_ESPG'
_3channel_precision_ESPG_old_postfix = '_3channel_precision_ESPG_old'
_1channel_precision_ESPG_postfix = '_1channel_precision_ESPG'
_3channel_precision_classifier_postfix = '_3channel_precision_classifier'
_3channel_precision_thres_classifier_postfix = '_3channel_precision_thres_classifier'
_3channel_resnet_postfix = '_3channel_resnet'
_1channel_precision_postfix = '_1channel_precision'
_5channel_precision_classifier_postfix = '_5channel_precision_classifier'
_5channel_resnet_postfix = '_5channel_resnet'
_3channel_precision_seg_postfix = '_3channel_precision_seg'
_new_2022_postfix = '_new_2022'
_new_2023_postfix = '_new_2023'
_3channel_precision_ESPG_viirsChanges_postfix = '_3channel_precision_ESPG_viirsChanges'

_3channel_precision_ESPG_viirsExtended_postfix = '_3channel_precision_ESPG_viirsExtended'

_new_west_postfix = '_new_west'
_new_east_postfix = '_new_east'
_everything_postfix = '_everything'
_th_neg_WITHOUTNEG_regression_postfix = '_th_neg_WITHOUTNEG_regression'
_everything_classifier_postfix = '_everything_classifier'
_everything_THpOS_classifier_postfix = '_everything_THpOS_classifier'
_workingSet_postfix = '_workingSet'
_workingSet_newslidingWindow_postfix = '_workingSet_newslidingWindow'

_workingSet_THNEG_regression_postfix = '_workingSet_THNEG_regression'

_pos_TH_classifier_postfix = '_pos_TH_classifier'
_pos_TH_neg_classifier_postfix = '_pos_TH_neg_classifier'
_pos_neg_balanced_classifier_postfix = '_pos_neg_balanced_classifier'
_TH_neg_balanced_classifier_postfix = '_TH_neg_balanced_classifier'
_paper_results_postfix= '_paper_results'
_temp_projection_check_postfix = '_temp_projection_check'
_everything_THpOS_usingposonly_classifier_postfix = '_everything_THpOS_usingposonly_classifier'
_everything_closeDate_postfix = '_everything_closeDate'
# site_Postfix = no_postfix
# referenceDir_speficic_Postfix = _temp_projection_check_postfix
# _everything_closeDate_collapsed_postfix = '_everything_closeDate_collapsed'
_everything_closeDate_correction_postfix = '_everything_closeDate_correction'
_everything_closeDate_correction_th_postfix = '_everything_closeDate_correction_th'
_everything_closeDate_correction_th_realtime_postfix = '_everything_closeDate_correction_th_realtime'

_everything_closeDate_correction_full_postfix = '_everything_closeDate_correction_full'
_everything_closeDate_correction_th_pos_postfix = '_everything_closeDate_correction_th_pos'
_everything_closeDate_correction_th_40_300_pos_postfix = '_everything_closeDate_correction_th_40_300_pos'
_everything_closeDate_correction_th_40_300_pos_small_postfix = '_everything_closeDate_correction_th_40_300_pos_small'
_everything_parallex_correction_postfix = '_everything_parallex_correction'

_everything_closeDate_correction_th_60_600_pos_postfix = '_everything_closeDate_correction_th_60_600_pos'


_realtime_api_check_postfix = '_realtime_api_check'




#training postfixs
site_Postfix = no_postfix
referenceDir_speficic_Postfix = _everything_closeDate_correction_postfix

trainingDir_speficic_Postfix = _everything_closeDate_correction_th_60_600_pos_postfix
model_specific_postfix = _everything_closeDate_correction_th_60_600_pos_postfix

result_specific_postfix = _3channel_precision_ESPG_postfix


# real time model
realtime_model_specific_postfix = _everything_closeDate_correction_th_realtime_postfix


gf_c_fields = [f'gf_c{i+1}' for i in range(GOES_Bands)]
training_data_field_names = ['vf'] + gf_c_fields + ['vf_FRP', 'gf_min', 'gf_max', 'vf_max']

# GOES_UNITS = 'Radiance'
GOES_UNITS = 'Brightness Temperature'
PREDICTION_UNITS = 'Brightness Temperature'
RES_OPT_PTH = 'SuperRes_Opt.pth'
RES_DECODER_PTH = 'SuperRes_Decoder.pth'
RES_ENCODER_PTH = 'SuperRes_Encoder.pth'
RES_AUTOENCODER_PTH = 'SuperRes_AutoEncoder.pth'
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
# data_dir = "DataRepository/reference_data"
compare = 'compare'

# data loading and preprocessing
# site_conf = 'config/configuration_2019.yml'
# toExecuteSiteList = "config/training_sites_2019"
site_conf = 'config/conf_sites.yml'
toExecuteSiteList = f"config/training_sites{site_Postfix}"
# toExecuteSiteList = "config/testing_sites"
# training = 'training'
# reference_data = "reference_data_working"

DataRepository = 'DataRepository'
reference_data = f"{DataRepository}/reference_data{referenceDir_speficic_Postfix}"
compare_dir = f'{reference_data}/compare/$LOC/'
# compare_dir = f'{reference_data}/compare_all/'
viirs_dir = f'{reference_data}/$LOC/VIIRS/'
goes_dir = f'{reference_data}/$LOC/GOES/$PROD_BAND/tif/'
training_dir = f'{DataRepository}/training_data{trainingDir_speficic_Postfix}/'
# training_dir = 'training_data_working/'
GOES_OVERWRITE = False
VIIRS_OVERWRITE = False

# "GOES-" + str(fire_date) + "_" + str(ac_time) + '.tif'
# self.satellite + '-' + str(fire_date) + "_" + str(ac_time) + '.tif'
GOES_tiff_file_name = 'GOES-{fire_date}_{ac_time}.tif'
VIIRS_tiff_file_name = 'viirs-snpp-{fire_date}_{ac_time}.tif'

# Autoencoder training and testing
# model_path = 'Model_BEFORE_MOVING_NORMALIZATION/'
model_path = 'Model/'
project_name_template = "{model_name}_{loss_function_name}_{n_epochs}epochs_{batch_size}batchsize_{learning_rate}lr{model_specific_postfix}"
test_split = 0.2
validation_split = 0.2
Results = f'{DataRepository}/results{result_specific_postfix}/'
# THRESHOLD_COVERAGE = 0.2
# THRESHOLD_IOU = 0.05
THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.453186035,0.005117899


# toExecuteSiteList = "config/testing_sites"
testing_dir = f'{DataRepository}/testing_dir/'
# realtimeSiteList = "config/realtime_sites"
RealTimeIncoming_files = f'{DataRepository}/RealTimeIncoming_files/$LOC/$RESULT_TYPE/'
RealTimeIncoming_results = f'{DataRepository}/RealTimeIncoming_results/$LOC/$RESULT_TYPE/'
validate_with_radar = False
videos = f'{DataRepository}/Videos/'

# blind testing
realtimeSiteList = "config/blind_testing_sites"

paper_results = ['713','122','956','728','118','553','408','387','849','104','663','609']
NO_SAMPLES = []
RANDOM_SAMPLES = [str(i) for i in range(7000) if i % 50 == 0]
ALL_SAMPLES = 0
SELECTED_SAMPLES = NO_SAMPLES


# if filename[0] in ['79', '126', '199', '729', '183', '992', '140', '189', '1159', '190', '26', '188']:
# if filename[0] in ['78','240','249','0','6','19','2','10','14','15','27','807']:
# if filename[0] in ['401','237','122','713','792','821','888','358','728','626','943','594','969','118','395','730','444','408','387','204','296','774','93','882','720','823','280','859','809','115','952','849','956','884','156','171','104','663','396']:
# if filename[0] in ['713','122','956','728','118','553','408','387','849','104','663','609']:
# if filename[0] in ['24']
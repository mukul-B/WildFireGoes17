# -*- coding: utf-8 -*-
"""
This script will run through the directory of training images, load
the image pairs, and then batch them before feading them into a pytorch based
autoencoder using MSE reconstruction loss for the superresolution.

Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from EvaluationMetricsAndUtilities import IOU_numpy, PSNR_intersection, PSNR_union, best_threshold_iteration, denoralize, getth, noralize_goes_to_radiance, noralize_viirs_to_radiance
from PlotInputandResults import ImagePlot, plot_from_ImagePlot


Prediction_JACCARD_LABEL = 'Prediction(Jaccard)'
Prediction_RMSE_LABEL = 'Prediction(RMSE)'
VIIRS_GROUND_TRUTH_LABEL = 'VIIRS Ground Truth'
OTSU_thresholding_on_GOES_LABEL = 'OTSU thresholding on GOES'
GOES_input_LABEL = 'GOES input'
plt.style.use('plot_style/wrf')
from GlobalValues import ALL_SAMPLES, GOES_UNITS, HC, HI, LI, LC, SELECTED_SAMPLES, THRESHOLD_COVERAGE, THRESHOLD_IOU, VIIRS_MIN_VAL, VIIRS_UNITS, GOES_Bands

class EvaluationVariables:
    def __init__(self,type):
        self.type = type
        self.iteration, self.ret, self.th = 0, 0, None
        self.th_l1 = None
        self.th_img = None
        self.iou = 0
        self.psnr_intersection = 0
        self.psnr_union = 0
        self.coverage = 0
        self.imagesize = 0
        self.dis = None

class EvaluateSingle:
    def __init__(self,coverage_converage,IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union, type):
        self.coverage_converage = coverage_converage
        self.IOU_control = IOU_control
        self.psnr_control_inter = psnr_control_inter
        self.psnr_control_union = psnr_control_union
        self.IOU_predicted = IOU_predicted
        self.psnr_predicted_inter = psnr_predicted_inter
        self.psnr_predicted_union = psnr_predicted_union
        
        self.type = type

def get_evaluation_results(prediction_rmse, prediction_IOU, inp, groundTruth, path, site, gf_min, gf_max, vf_max, LOSS_NAME, frp = None):

    pl = path.split('/')
    filename = pl[-1].split('.')
    site_date_time = '_'.join(site.split('.')[1].split('_')).replace(' ','_' ).replace('-','_' )

    # groundTruth_NZ_min = groundTruth[groundTruth>0].min()
    # groundTruth_max = groundTruth.max()

    # 1)Input 
    inp = inp.numpy()

    # 2)Ground truth
    groundTruth = groundTruth.numpy()
    frp = frp.numpy()

    # 3)Evaluation on Input after OTSU thresholding
    inputEV = EvaluationVariables("input")
    # extract_img = inp
    extract_img = inp if GOES_Bands == 1 else inp[0]
    inputEV.iteration, inputEV.ret, inputEV.th = best_threshold_iteration(groundTruth, extract_img)
    _, inputEV.th_l1, _, _, _ = getth(extract_img, on=0)
    inputEV.th_img = inputEV.th * extract_img
    inputEV.iou = IOU_numpy(groundTruth, inputEV.th)
    inputEV.psnr_intersection = PSNR_intersection(groundTruth, inputEV.th_img)
    inputEV.psnr_union = PSNR_union(groundTruth, inputEV.th_img)
    inputEV.coverage = np.count_nonzero(inputEV.th_l1) / inputEV.th.size
    inputEV.imagesize = extract_img.size
    # inputEV.dis = ''
    inputEV.dis = f'\nThreshold (Iteration:{str(inputEV.iteration)}): {str(round(inputEV.ret, 4))} Coverage: {str(round(inputEV.coverage, 4))}' \
                f'\nIOU : {str(inputEV.iou)}' \
                f'\nPSNR_intersection : {str(round(inputEV.psnr_intersection, 4))}'

     #  7)rmse prediction evaluation
    if(LOSS_NAME == 'Classification_loss'):
        predRMSEEV = EvaluationVariables("prediction_Classification")
        if prediction_rmse is not None:
            # outmap_min = prediction_rmse.min()
            # outmap_max = prediction_rmse.max()
            # prediction_rmse_normal = (prediction_rmse - outmap_min) / (outmap_max - outmap_min)

            # prediction_rmse = prediction_rmse.numpy()
            zx = 1 if np.sum(groundTruth) > 0 else 0
            predRMSEEV.TP = np.sum(zx == 1 and zx == prediction_rmse.item())
            predRMSEEV.FN = np.sum(zx == 1 and zx != prediction_rmse.item())
            predRMSEEV.FP = np.sum(zx == 0 and zx != prediction_rmse.item())
            predRMSEEV.TN = np.sum(zx == 0 and zx == prediction_rmse.item())

            predRMSEEV.iou = np.sum(zx == 1 and zx == prediction_rmse.item())
            predRMSEEV.psnr_intersection = np.sum(zx == 1 and zx != prediction_rmse.item())
            predRMSEEV.psnr_union = np.sum(zx == 0 and zx != prediction_rmse.item())
            predRMSEEV.coverage = np.sum(zx == 0 and zx == prediction_rmse.item())
            
            TRUEFalse= HC if np.sum(zx == prediction_rmse.item()) > 0 else  LC
            positiveNegitive = HI if prediction_rmse.item() == 1 else LI
            condition = TRUEFalse + positiveNegitive
            predRMSEEV.dis = f''

    else:
        # finding coverage and intensity criteria based on input and groundthruth
        condition_coverage = HC if inputEV.coverage > THRESHOLD_COVERAGE else LC
        condition_intensity = HI if inputEV.iou > THRESHOLD_IOU else LI
        condition = condition_coverage + condition_intensity
        

        #  4)rmse prediction evaluation
        predRMSEEV = EvaluationVariables("prediction_rmse")
        if prediction_rmse is not None:
            outmap_min = prediction_rmse.min()
            outmap_max = prediction_rmse.max()
            prediction_rmse_normal = (prediction_rmse - outmap_min) / (outmap_max - outmap_min)

            # prediction_rmse = prediction_rmse.numpy()
            prediction_rmse = prediction_rmse_normal.numpy()
            prediction_rmse = np.nan_to_num(prediction_rmse)
            
            predRMSEEV.ret, predRMSEEV.th, histogram,_, _ = getth(prediction_rmse, on=0)
            # retN, thN, _,_, _ = getth(prediction_rmse_normal, on=0)

            # prediction_rmse_TH = thN * prediction_rmse_normal
            
            predRMSEEV.th_img = predRMSEEV.th * prediction_rmse
            predRMSEEV.iou = IOU_numpy(groundTruth, predRMSEEV.th)
            predRMSEEV.psnr_intersection = PSNR_intersection(groundTruth, predRMSEEV.th_img)
            predRMSEEV.psnr_union = PSNR_union(groundTruth, predRMSEEV.th_img)
            predRMSEEV.coverage = np.count_nonzero(predRMSEEV.th) / predRMSEEV.th.size
            predRMSEEV.dis = f'\nThreshold: {str(round(predRMSEEV.ret, 4))}  Coverage:  {str(round(predRMSEEV.coverage, 4))} ' \
                    f'\nIOU :  {str(predRMSEEV.iou)} ' \
                    f'\nPSNR_intersection : {str(round(predRMSEEV.psnr_intersection, 4))}'

        # 5)IOU prediction evaluation
        predIOUEV = EvaluationVariables("prediction_jaccard")
        if prediction_IOU is not None:
            prediction_IOU = prediction_IOU.numpy()
            predIOUEV.ret, predIOUEV.th,  _,_, _ = getth(prediction_IOU, on=0)
            predIOUEV.th_img = predIOUEV.th * prediction_IOU
            predIOUEV.iou = IOU_numpy(groundTruth, predIOUEV.th)
            predIOUEV.coverage = np.count_nonzero(predIOUEV.th) / predIOUEV.th.size
            predIOUEV.dis = f'\nThreshold:  {str(round(predIOUEV.ret, 4))}  Coverage:  {str(round(predIOUEV.coverage, 4))} ' \
                    f'\nIOU :  {str(predIOUEV.iou)}'

    (iou_p, psnr_intersection_p, psnr_union_p) = (predRMSEEV.iou, predRMSEEV.psnr_intersection, predRMSEEV.psnr_union) if (
            prediction_rmse is not None) else (predIOUEV.iou, 0, 0)
    
    eval_single = EvaluateSingle(predRMSEEV.coverage, inputEV.iou, inputEV.psnr_intersection, inputEV.psnr_union, iou_p, psnr_intersection_p, psnr_union_p, condition) 
    # logging.info(f'{inputEV.iteration},{filename[0]},{condition}')
    
    # ----------------------------------------------------------------------------------
    # random result plot
    if ALL_SAMPLES or filename[0] in SELECTED_SAMPLES :
    # if condition in (LC+HI, LC+LI ) :
    # if 0:
    # if np.count_nonzero(groundTruth) > 100:

        
        g1 = ImagePlot(GOES_UNITS,gf_max, gf_min,
                       extract_img, 
                       GOES_input_LABEL)
        
        

        # g2 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                inputEV.th_l1 * inp,
        #                OTSU_thresholding_on_GOES_LABEL + inputEV.dis)
        # minmaxg3 = "\n second min: " + str(groundTruth_NZ_min) +"\n max: " + str(groundTruth_max)
        g3 = ImagePlot(VIIRS_UNITS,vf_max,VIIRS_MIN_VAL,
                       groundTruth ,
                       VIIRS_GROUND_TRUTH_LABEL + str(np.count_nonzero(groundTruth)) + ' ' +str(np.sum(frp) ))
        # g6 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                predIOUEV.th_img,
        #                VIIRS_GROUND_TRUTH_LABEL + minmaxg3)

        
        
        if(LOSS_NAME == 'Classification_loss'):
            # classification
            g4 = ImagePlot(None,vf_max,VIIRS_MIN_VAL,
                        prediction_rmse if prediction_rmse is not None else prediction_IOU,
                        str(prediction_rmse.item())+' '+Prediction_RMSE_LABEL if prediction_rmse is not None else Prediction_JACCARD_LABEL)
        else:
            # minmaxg4 = "\n second min: " + str(outmap_NZ_min) +"\n max: " + str(outmap_max) +"\n psnr_intersection: " + str(round(predRMSEEV.psnr_intersection,4))+"\n psnr_union: " + str(round(predRMSEEV.psnr_union,4)) + "\nIOU: " +str(round(predRMSEEV.iou,4))
            g4 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else "IOU",vf_max,VIIRS_MIN_VAL,
                        prediction_rmse if prediction_rmse is not None else prediction_IOU,
                        Prediction_RMSE_LABEL+ pl[1] if prediction_rmse is not None else Prediction_JACCARD_LABEL)

        
        # psnr_intersection6 = PSNR_intersection(groundTruth, thN * prediction_rmse_TH)
        # psnr_union6 = PSNR_union(groundTruth, thN * prediction_rmse_TH)
        # IOU6 = IOU_numpy(groundTruth, thN * prediction_rmse_TH)
        
        # minmaxg6 = "\n psnr_intersection: " + str(round(psnr_intersection6,4))+"\n psnr_union: " + str(round(psnr_union6,4)) + "\nIOU: " +str(round(IOU6,4))
        # g6 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else "IOU",vf_max,VIIRS_MIN_VAL,
        #                prediction_rmse_TH if prediction_rmse is not None else prediction_IOU,
        #                "thresholded: "+ str(normalth) +"\n" + Prediction_RMSE_LABEL + minmaxg6 if prediction_rmse is not None else Prediction_JACCARD_LABEL)
        
        
        # g5 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else None,vf_max,VIIRS_MIN_VAL,
        #                th_img_rmse if prediction_rmse is not None else None,
        #                'OTSU thresholding on Prediction(RMSE)' + ret1_dis)
        # g6 = ImagePlot(VIIRS_UNITS if prediction_IOU is not None else None,vf_max,VIIRS_MIN_VAL,
        #                th3_img if prediction_IOU is not None else None,
        #                'OTSU thresholding on Prediction(IOU)' + ret3_dis)
        img_seq = ((g1,g3,g4),)
        if(len(inp) > 1):
            active_fire = 100 *( (inp[0]-inp[1])/(inp[0]+inp[1]) )
            # gf_min, gf_max = [210, 207, 205],[413,342, 342]
            cloud_mask_activeFire = denoralize(inp[2],342, 205)
            cloud_mask_activeFire = cloud_mask_activeFire * (cloud_mask_activeFire > 280)
            
           
            
            g7 = ImagePlot(GOES_UNITS,np.max(active_fire), np.min(active_fire),
                        active_fire, 
                        "Active fire: normalized diff band7 ,band 14")
            
            g8 = ImagePlot(GOES_UNITS,342, 205,
                        cloud_mask_activeFire, 
                        "Band15 > 280")
            
            g9 = ImagePlot(GOES_UNITS,np.max(active_fire), np.min(active_fire),
                        active_fire * (cloud_mask_activeFire > 280), 
                        "Cloud mask Active fire: band15 > 280")
            img_seq = ((g1,g7,g8),(g3,g4,g9))
        
        # img_seq = ((g1,g3,g4,g5,g6),)
        # img_seq = ((g1,g3),)
        # path ='/'.join(pl[:-1]+[f'{str(round(iou_p,4))}_{str(round(inputEV.coverage,4))}_{str(round(inputEV.iou,4))}_{pl[-1]}']) 
        plot_from_ImagePlot(site_date_time+ pl[1],img_seq,condition,path)
        
        # print("-----------------------------------------------------------------------")
        logging.info(
        f'{LOSS_NAME},{condition},{filename[0]},{str(predRMSEEV.iou) if prediction_rmse is not None else ""},{str(predIOUEV.iou) if prediction_IOU is not None else ""},{predRMSEEV.psnr_intersection if prediction_rmse is not None else ""},{site_date_time}')
    # logging.info(
    #     f'{condition},{coverage_i},{iou_i},{condition_coverage},{condition_intensity}')
       
    return eval_single

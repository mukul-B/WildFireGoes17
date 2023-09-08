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
from EvaluationOperations import IOU_numpy, PSNR_intersection, PSNR_union, best_threshold_iteration, getth, noralize_goes_to_radiance, noralize_viirs_to_radiance


Prediction_JACCARD_LABEL = 'Prediction(Jaccard)'
Prediction_RMSE_LABEL = 'Prediction(RMSE)'
VIIRS_GROUND_TRUTH_LABEL = 'VIIRS Ground Truth'
OTSU_thresholding_on_GOES_LABEL = 'OTSU thresholding on GOES'
GOES_input_LABEL = 'GOES input'
plt.style.use('plot_style/wrf')
from GlobalValues import GOES_UNITS, HC, HI, LI, LC, THRESHOLD_COVERAGE, THRESHOLD_IOU, VIIRS_MIN_VAL, VIIRS_UNITS


class ImagePlot:
    def __init__(self,unit,vmax,vmin,image_blocks,lable_blocks):
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        if(self.unit):
            self.image_blocks = noralize_goes_to_radiance(image_blocks,vmax,vmin) if unit==GOES_UNITS else noralize_viirs_to_radiance(image_blocks,vmax)
        else:
            self.image_blocks = None
        self.lable_blocks = lable_blocks

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
def plot_histogramandimage(image,path):
    # image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    # histogram, bins = np.histogram(image, bins=256, range=(0, 256))
    ret2, th2, hist2, bins2, index_of_max_val2 = getth(image)

    

    # Create subplots with constrained layout
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 8))

    # Plot the image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Image")
    axs[0].axis('off')

    # Plot the histogram
    # axs[1].plot(hist2, color='black')
    axs[1].hist(bins2[1:-1], bins=bins2[1:], weights=hist2[1:], color='blue')
    axs[1].set_title("Histogram")
    axs[1].set_xlabel("Pixel Value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 255)
    # axs[1].set_ylim(0, 200)

    # show_hist([1], "Histogram", bins2, hist2, 1)
    

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def safe_results(prediction_rmse, prediction_IOU, inp, groundTruth, path, site, gf_min, gf_max, vf_max, LOSS_NAME):

    pl = path.split('/')
    filename = pl[-1].split('.')

    groundTruth_NZ_min = groundTruth[groundTruth>0].min()
    groundTruth_max = groundTruth.max()

    # 1)Input 
    inp = inp.numpy()

    # 2)Ground truth
    groundTruth = groundTruth.numpy()

    # 3)Evaluation on Input after OTSU thresholding
    inputEV = EvaluationVariables("input")
    inputEV.iteration, inputEV.ret, inputEV.th = best_threshold_iteration(groundTruth, inp)
    _, inputEV.th_l1, _, _, _ = getth(inp, on=0)
    inputEV.th_img = inputEV.th * inp
    inputEV.iou = IOU_numpy(groundTruth, inputEV.th)
    inputEV.psnr_intersection = PSNR_intersection(groundTruth, inputEV.th_img)
    inputEV.psnr_union = PSNR_union(groundTruth, inputEV.th_img)
    inputEV.coverage = np.count_nonzero(inputEV.th_l1) / inputEV.th.size
    inputEV.imagesize = inp.size
    inputEV.dis = ''
    inputEV.dis = f'\nThreshold (Iteration:{str(inputEV.iteration)}): {str(round(inputEV.ret, 4))} Coverage: {str(round(inputEV.coverage, 4))}' \
                f'\nIOU : {str(inputEV.iou)}' \
                f'\nPSNR_intersection : {str(round(inputEV.psnr_intersection, 4))}'

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
    predIOUEV = EvaluationVariables("prediction_rmse")
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
    
    # logging.info(f'{inputEV.iteration},{filename[0]},{condition}')
    
    # ----------------------------------------------------------------------------------
    # random result plot

    # if filename[0] in ['79', '126', '199', '729', '183', '992', '140', '189', '1159', '190', '26', '188']:
     # if filename[0] in ['78','240','249','0','6','19','2','10','14','15','27','807']:
    # if filename[0] in ['401','237','122','713','792','821','888','358','728','626','943','594','969','118','395','730','444','408','387','204','296','774','93','882','720','823','280','859','809','115','952','849','956','884','156','171','104','663','396']:
    if filename[0] in ['713','122','956','728','118','358','408','387','849','104','663','609']:
    # if inputEV.iteration > 1 :
    # if 1:
    # if filename[0] in ['24']:
        g1 = ImagePlot(GOES_UNITS,gf_max, gf_min,
                       inp, 
                       GOES_input_LABEL)
        # g2 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                inputEV.th_l1 * inp,
        #                OTSU_thresholding_on_GOES_LABEL + inputEV.dis)
        # minmaxg3 = "\n second min: " + str(groundTruth_NZ_min) +"\n max: " + str(groundTruth_max)
        g3 = ImagePlot(VIIRS_UNITS,vf_max,VIIRS_MIN_VAL,
                       groundTruth ,
                       VIIRS_GROUND_TRUTH_LABEL )
        # g6 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                predIOUEV.th_img,
        #                VIIRS_GROUND_TRUTH_LABEL + minmaxg3)

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
        # img_seq = ((g1,g3),(g4,g6))
        # img_seq = ((g1,g3,g4,g5,g6),)
        # img_seq = ((g1,g3),)
        # path ='/'.join(pl[:-1]+[f'{str(round(iou_p,4))}_{str(round(inputEV.coverage,4))}_{str(round(inputEV.iou,4))}_{pl[-1]}']) 
        plot_it(img_seq,condition,path)
        site_date_time = '_'.join(site.split('.')[1].split('_'))
        # print("-----------------------------------------------------------------------")
        logging.info(
        f'{LOSS_NAME},{condition},{filename[0]},{str(predRMSEEV.iou) if prediction_rmse is not None else ""},{str(predIOUEV.iou) if prediction_IOU is not None else ""},{predRMSEEV.psnr_intersection if prediction_rmse is not None else ""},{site_date_time}')
    # logging.info(
    #     f'{condition},{coverage_i},{iou_i},{condition_coverage},{condition_intensity}')
        
    return inputEV.coverage, inputEV.iou, inputEV.psnr_intersection, inputEV.psnr_union, iou_p, psnr_intersection_p, psnr_union_p, condition

def plot_it(img_seq,condition,path,colection=False):
    pl = path.split('/')
    # filename = pl[-1].split('.')
    filename = pl[-1].replace('.png','')
    c,r = len(img_seq) ,len(img_seq[0])
    if(colection):
        fig, axs = plt.subplots(c, r, constrained_layout=True, figsize=(12, 4*c))
    for col in range(c):
        for row in range(r):
            
            image_blocks=img_seq[col][row].image_blocks
            lable_blocks=img_seq[col][row].lable_blocks
            cb_unit=img_seq[col][row].unit
            # vmin,vmax = img_seq[col][row].vmin, img_seq[col][row].vmax
            vmin,vmax = 0 , 413
            if(colection):
                if(r == 1):
                    ax = axs[col]
                elif(c==1):
                    ax = axs[row]
                else:
                    ax = axs[col][row]
                ax.set_title(lable_blocks)

            if  image_blocks is not None:
                if(not colection):
                    fig = plt.figure()
                    ax = fig.add_subplot()

                X, Y = np.mgrid[0:1:128j, 0:1:128j]
                
                # vmin = VIIRS_MIN_VAL if lable_blocks in [VIIRS_GROUND_TRUTH] else None
                # vmax =420

                # cb_unit = "Background | Fire Area     " if lable_blocks in [
                #     Prediction_JACCARD] else VIIRS_UNITS
                if lable_blocks in [Prediction_JACCARD_LABEL]:
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r", 2), vmin=vmin, vmax=vmax)
                else:
                    # sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r"), vmin=vmin, vmax=vmax)
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("jet"), vmin=vmin, vmax=vmax)
                    cb = fig.colorbar(sc, pad=0.01,ax=ax)
                    cb.ax.tick_params(labelsize=11)
                    # cb.set_label(cb_unit, fontsize=12)
                    cb.set_label(cb_unit, fontsize=12)
                
                # plt.tick_params(left=False, right=False, labelleft=False,
                #                 labelbottom=False, bottom=False)
                # Hide X and Y axes label marks
                # ax.xaxis.set_tick_params(labelbottom=False)
                # ax.yaxis.set_tick_params(labelleft=False)
                
                # Hide X and Y axes tick marks
                ax.set_xticks([])
                ax.set_yticks([])
                if(not colection):
                    filenamecorrection = lable_blocks.replace('\n','_').replace(' ','_').replace('.','_').replace('(','_').replace(')','_').replace(':','_')
                    # path ='/'.join(pl[:2]) + f'/{condition}_{filename}_{filenamecorrection}.png'
                    path = '/'.join(pl[:1]) + f"/allresults/{condition}/{filename}_{filenamecorrection}.png" 
                    print(path)
                    plt.savefig(path,
                                bbox_inches='tight', dpi=600)
                    plt.show()
                    plt.close()
    if(colection):
        # path ="check.png"
        path = '/'.join(pl[:1]) + f"allresults/{condition}/{filename}/{pl[:1]}.png" 
        # path = "cheko.png"
        # path = '/'.join(pl[:2]) + f"/{condition}/{filename[0]}_{output_iou}_{psnr_intersection_i}_{psnr_union_i}_{str(round(coverage_i, 4))}.png"
        plt.rcParams['savefig.dpi'] = 600
        fig.savefig(path)
        # input()
        
        # plt.show()
        plt.close()
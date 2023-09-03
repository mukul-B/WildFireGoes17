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
import math

import matplotlib.pyplot as plt
import numpy as np


Prediction_JACCARD_LABEL = 'Prediction(Jaccard)'
Prediction_RMSE_LABEL = 'Prediction(RMSE)'
VIIRS_GROUND_TRUTH_LABEL = 'VIIRS Ground Truth'
OTSU_thresholding_on_GOES_LABEL = 'OTSU thresholding on GOES'
GOES_input_LABEL = 'GOES input'
plt.style.use('plot_style/wrf')
from GlobalValues import GOES_UNITS, HC, HI, LI, LC, THRESHOLD_COVERAGE, THRESHOLD_IOU, VIIRS_MIN_VAL, VIIRS_UNITS


# logging.basicConfig(filename='evaluation.log', filemode='w',format='%(message)s', level=logging.INFO)


def getth(image, bin_edges, on=0):
    # Set total number of bins in the histogram
    image_r = image.copy()
    image_r = image_r * 255
    # Get the image histogram
    hist, bin_edges = np.histogram(image_r, bins=256)
    if (on):
        hist, bin_edges = hist[on:], bin_edges[on:]
    # Get normalized histogram if it is required

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    threshold = threshold + on if ((threshold + on) < 255) else threshold
    image_r[image_r < threshold] = 0
    image_r[image_r >= threshold] = 1
    return round(threshold, 2), image_r, hist, bin_edges, index_of_max_val


def IOU_numpy(target, pred):
    pred_binary = np.copy(pred)
    target_binary = np.copy(target)
    pred_binary[pred_binary != 0] = 1
    target_binary[target_binary != 0] = 1
    intersection = (pred_binary * target_binary).sum()
    total = (pred_binary + target_binary).sum()
    union = total - intersection
    IOU = (intersection) / (union)
    # IOU = (intersection + SMOOTH) / (union + SMOOTH)
    return IOU


def PSNR_union(target, pred):
    imdff = pred - target
    # union = pred + target
    union = target
    imdff[union == 0] = 0
    imdff = imdff.flatten()
    pixcelsInUnion = np.count_nonzero(union)
    if pixcelsInUnion > 0:
        rmse = math.sqrt(np.sum(np.array(imdff ** 2)) / pixcelsInUnion)
    else:
        return 0
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def PSNR_intersection(target, pred):
    imdff = pred - target
    interaction = pred * target
    imdff[interaction == 0] = 0
    imdff =  imdff.flatten()
    pixelsInIntersection = np.count_nonzero(interaction)
    if pixelsInIntersection > 0:
        # return math.sqrt(np.sum(np.array(imdff ** 2)) / pixelsInIntersection)
        rmse = math.sqrt(np.sum(np.array(imdff ** 2)) / pixelsInIntersection)
    else:
        return 0
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def show_img(axis, img, label):
    # axis.imshow(img)
    axis.imshow(img, cmap='gray')
    axis.set_title(label, size=8)


def show_hist(axis, title, binsl, hist, minr_tick):
    axis.hist(binsl[:-1], bins=binsl, weights=hist, color='blue')
    # # axis.set_title(title, size=8)
    # axis.set_xticks(minr_tick, minor=True, color='red')
    # axis.set_xticklabels(minr_tick, fontdict=None, minor=True, color='red', size=13)
    axis.tick_params(axis='x', which='minor', colors='red', size=13)
    axis.set_ylabel("Pixels Fraction", fontsize=15)
    axis.set_xlabel('Normalized Radiance', fontsize=15)
    axis.set_xlim(0, )


def best_threshold_iteration(groundTruth, input):
    maxiou = 0
    level = 0
    pth2 = np.ones_like(input)
    pret2 = 0
    phist2, pbins2 = None, None
    imgs = []
    thr_list = []
    iou_list = []
    l = 0
    pcoverage = 0
    while (True):
        ret2, th2, hist2, bins2, index_of_max_val2 = getth(pth2 * input, 256, on=int(pret2))
        iou_i = IOU_numpy(groundTruth, th2)

        if maxiou >= iou_i:
            break
        maxiou = iou_i
        level += 1
        pret2, pth2, phist2, pbins2 = ret2, th2, hist2, bins2
    ret2, th2, hist2, bins2 = pret2, pth2, phist2, pbins2
    return level, ret2, th2


def noralize_goes_to_radiance(ngf, gf_max, gf_min):
    color_normal_value = 1
    return (gf_min + (ngf * ((gf_max - gf_min) / color_normal_value))).round(5)


def noralize_viirs_to_radiance(nvf, vf_max,vf_min=0):
    color_normal_value = 1
    return (vf_min + (nvf * ((vf_max-vf_min) / color_normal_value))).round(2)

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
    ret2, th2, hist2, bins2, index_of_max_val2 = getth(image, 256)

    

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

def safe_results(prediction_rmse, prediction_IOU, input, groundTruth, path, site, gf_min, gf_max, vf_max, LOSS_NAME):
    
    groundTruth_NZ_min = groundTruth[groundTruth>0].min()
    groundTruth_max = groundTruth.max()
    
    
    # iou_rmse = psnr_intersection_rmse = psnr_union_rmse = 0
    # 1)Input 
    input = input.numpy()

    # 2)Ground truth
    groundTruth = groundTruth.numpy()

    # 3)Evaluation on Input after OTSU thresholding

    inputEV = EvaluationVariables("input")
    inputEV.iteration, inputEV.ret, inputEV.th = best_threshold_iteration(groundTruth, input)
    _, inputEV.th_l1, _, _, _ = getth(input, 256, on=0)
    inputEV.th_img = inputEV.th * input
    inputEV.iou = IOU_numpy(groundTruth, inputEV.th)
    inputEV.psnr_intersection = PSNR_intersection(groundTruth, inputEV.th_img)
    inputEV.psnr_union = PSNR_union(groundTruth, inputEV.th_img)
    inputEV.coverage = np.count_nonzero(inputEV.th_l1) / inputEV.th.size
    inputEV.imagesize = input.size
    inputEV.dis = ''
    inputEV.dis = f'\nThreshold (Iteration:{str(inputEV.iteration)}): {str(round(inputEV.ret, 4))} Coverage: {str(round(inputEV.coverage, 4))}' \
                f'\nIOU : {str(inputEV.iou)}' \
                f'\nPSNR_intersection : {str(round(inputEV.psnr_intersection, 4))}'

    condition_coverage = HC if inputEV.coverage > THRESHOLD_COVERAGE else LC
    condition_intensity = HI if inputEV.iou > THRESHOLD_IOU else LI
    condition = condition_coverage + condition_intensity
    pl = path.split('/')
    filename = pl[-1].split('.')

    #  4)rmse prediction evaluation
    predRMSEEV = EvaluationVariables("prediction_rmse")
    if prediction_rmse is not None:
        outmap_min = prediction_rmse.min()
        outmap_NZ = prediction_rmse[prediction_rmse>0.01]
        if outmap_NZ.numel() > 0:
            outmap_NZ_min = outmap_NZ.min()
        else:
            outmap_NZ_min = 0
    
        outmap_max = prediction_rmse.max()
        
        # logging.info(f'{condition},{filename[0]},{outmap_NZ_min},{outmap_max},{groundTruth_NZ_min},{groundTruth_max}')
        if (outmap_max>0.18):
            
            prediction_rmse = (prediction_rmse - outmap_min) / (outmap_max - outmap_min)
        prediction_rmse = prediction_rmse.numpy()
        predRMSEEV.ret, predRMSEEV.th, histogram,_, _ = getth(prediction_rmse, 256, on=0)
        # print(histogram)
        # if filename[0] in ['79']:
        if 1:
         plot_histogramandimage(groundTruth,"results/histograms_groundtruth/"+filename[0]+".png")
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
        predIOUEV.ret, predIOUEV.th,  _,_, _ = getth(prediction_IOU, 256, on=0)
        predIOUEV.th_img = predIOUEV.th * prediction_IOU
        predIOUEV.iou = IOU_numpy(groundTruth, predIOUEV.th)
        predIOUEV.coverage = np.count_nonzero(predIOUEV.th) / predIOUEV.th.size
        predIOUEV.dis = f'\nThreshold:  {str(round(predIOUEV.ret, 4))}  Coverage:  {str(round(predIOUEV.coverage, 4))} ' \
                   f'\nIOU :  {str(predIOUEV.iou)}'

    (iou_p, psnr_intersection_p, psnr_union_p) = (predRMSEEV.iou, predRMSEEV.psnr_intersection, predRMSEEV.psnr_union) if (
            prediction_rmse is not None) else (predIOUEV.iou, 0, 0)
    
    
    
    # ----------------------------------------------------------------------------------
    # random result plot

    # if filename[0] in ['79', '126', '199', '729', '183', '992', '140', '189', '1159', '190', '26', '188']:
     # if filename[0] in ['78','240','249','0','6','19','2','10','14','15','27','807']:

    if 0:
    # if filename[0] in ['956']:
        g1 = ImagePlot(GOES_UNITS,gf_max, gf_min,
                       input, 
                       GOES_input_LABEL)
        # g2 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                th_img_i,
        #                OTSU_thresholding_on_GOES + input_dis)
        g3 = ImagePlot(VIIRS_UNITS,vf_max,VIIRS_MIN_VAL,
                       groundTruth,
                       VIIRS_GROUND_TRUTH_LABEL)
        g4 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else "IOU",vf_max,VIIRS_MIN_VAL,
                       prediction_rmse if prediction_rmse is not None else prediction_IOU,
                       Prediction_RMSE_LABEL if prediction_rmse is not None else Prediction_JACCARD_LABEL)
        # g5 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else None,vf_max,VIIRS_MIN_VAL,
        #                th_img_rmse if prediction_rmse is not None else None,
        #                'OTSU thresholding on Prediction(RMSE)' + ret1_dis)
        # g6 = ImagePlot(VIIRS_UNITS if prediction_IOU is not None else None,vf_max,VIIRS_MIN_VAL,
        #                th3_img if prediction_IOU is not None else None,
        #                'OTSU thresholding on Prediction(IOU)' + ret3_dis)
        # img_seq = ((g1,g2,g3),(g4,g5,g6))
        # img_seq = ((g1,g3),(g4,g6))
        img_seq = ((g1,g3,g4),)
        # img_seq = ((g1,g3),)
        plot_it(img_seq,condition,path)
        site_date_time = '_'.join(site.split('.')[1].split('_'))
        # print("-----------------------------------------------------------------------")
        logging.info(
        f'{LOSS_NAME},{condition},{filename[0]},{str(predRMSEEV.iou) if prediction_rmse is not None else ""},{str(predIOUEV.iou) if prediction_IOU is not None else ""},{predRMSEEV.psnr_intersection if prediction_rmse is not None else ""},{site_date_time}')
    # logging.info(
    #     f'{condition},{coverage_i},{iou_i},{condition_coverage},{condition_intensity}')
        
    return inputEV.coverage, inputEV.iou, inputEV.psnr_intersection, inputEV.psnr_union, iou_p, psnr_intersection_p, psnr_union_p, condition

def plot_it(img_seq,condition,path,colection=True):
    pl = path.split('/')
    filename = pl[-1].split('.')
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
                    # plt.savefig('/'.join(pl[:2]) + f'/{condition}_{filename[0]}_{filenamecorrection}.png',
                    #             bbox_inches='tight', dpi=600)
                    plt.show()
                    plt.close()
    if(colection):
        # path ="check.png"
        path = '/'.join(pl[
                        :2]) + f"/{condition}/{filename[0]}.png" 
        # path = '/'.join(pl[
        #                 :2]) + f"/{condition}/{filename[0]}_{output_iou}_{psnr_intersection_i}_{psnr_union_i}_{str(round(coverage_i, 4))}.png"
        plt.rcParams['savefig.dpi'] = 600
        fig.savefig(path)
        # plt.show()
        plt.close()
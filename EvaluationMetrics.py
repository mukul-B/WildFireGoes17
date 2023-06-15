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

Prediction_JACCARD = 'Prediction(Jaccard)'
Prediction_RMSE = 'Prediction(RMSE)'
VIIRS_GROUND_TRUTH = 'VIIRS Ground Truth'
OTSU_thresholding_on_GOES = 'OTSU thresholding on GOES'
GOES_input = 'GOES input'
plt.style.use('plot_style/wrf')
from GlobalValues import GOES_UNITS, HC, HI, LI, LC, VIIRS_MIN_VAL, VIIRS_UNITS


# logging.basicConfig(filename='evaluation.log', filemode='w',format='%(message)s', level=logging.INFO)


def getth(image, bin_edges, on=0):
    # Set total number of bins in the histogram
    image_r = image.copy()
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
    union = pred + target
    imdff[union == 0] = 0
    imdff = (1 / 255) * imdff.flatten()
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
    imdff = (1 / 255) * imdff.flatten()
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
    # axis.set_title(title, size=8)
    axis.set_xticks(minr_tick, minor=True, color='red')
    axis.set_xticklabels(minr_tick, fontdict=None, minor=True, color='red', size=13)
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
    color_normal_value = 255
    return (gf_min + (ngf * ((gf_max - gf_min) / color_normal_value))).round(5)


def noralize_viirs_to_radiance(nvf, vf_max,vf_min=0):
    color_normal_value = 255
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
        
def safe_results(prediction_rmse, prediction_IOU, input, groundTruth, path, site, gf_min, gf_max, vf_max, LOSS_NAME):
    
    iou_rmse = psnr_intersection_rmse = psnr_union_rmse = 0
    # 1)Input 
    # ret2, th2, hist2, bins2, index_of_max_val2 = getth(input, 256, on=0)
    input = input.numpy() * 255

    # 2)Ground truth
    groundTruth = groundTruth.numpy() * 255

    # 3)Evaluation on Input after OTSU thresholding
    iteration, ret2, th2 = best_threshold_iteration(groundTruth, input)
    _, th_l1, _, _, _ = getth(input, 256, on=0)
    th_img_i = th2 * input
    iou_i = IOU_numpy(groundTruth, th2)
    psnr_intersection_i = PSNR_intersection(groundTruth, th_img_i)
    psnr_union_i = PSNR_union(groundTruth, th_img_i)
    coverage_i = np.count_nonzero(th_l1) / th2.size
    imagesize = input.size
    input_dis = ''
    input_dis = f'\nThreshold (Iteration:{str(iteration)}): {str(round(ret2, 4))} Coverage: {str(round(coverage_i, 4))}' \
                f'\nIOU : {str(iou_i)}' \
                f'\nPSNR_intersection : {str(round(psnr_intersection_i, 4))}'

    #  4)rmse prediction evaluation
    output_iou = None
    ret1_dis, ret3_dis = '', ''
    if prediction_rmse is not None:
        prediction_rmse = prediction_rmse.numpy() * 255
        ret1, th1, hist1, bins1, index_of_max_val1 = getth(prediction_rmse, 256, on=0)
        th_img_rmse = th1 * prediction_rmse
        iou_rmse = IOU_numpy(groundTruth, th1)
        psnr_intersection_rmse = PSNR_intersection(groundTruth, th_img_rmse)
        psnr_union_rmse = PSNR_union(groundTruth, th_img_rmse)
        coverage_p = np.count_nonzero(th1) / th1.size
        output_iou = iou_rmse
        ret1_dis = f'\nThreshold: {str(round(ret1, 4))}  Coverage:  {str(round(coverage_p, 4))} ' \
                   f'\nIOU :  {str(iou_rmse)} ' \
                   f'\nPSNR_intersection : {str(round(psnr_intersection_rmse, 4))}'

    # 5)IOU prediction evaluation
    if prediction_IOU is not None:
        prediction_IOU = prediction_IOU.numpy() * 255
        ret3, th3, hist3, bins3, index_of_max_val3 = getth(prediction_IOU, 256, on=0)
        th3_img = th3 * prediction_IOU
        iou_p_iou = IOU_numpy(groundTruth, th3)
        coverage_p_iou = np.count_nonzero(th3) / th3.size
        output_iou = iou_p_iou
        ret3_dis = f'\nThreshold:  {str(round(ret3, 4))}  Coverage:  {str(round(coverage_p_iou, 4))} ' \
                   f'\nIOU :  {str(iou_p_iou)}'

    (iou_p, psnr_intersection_p, psnr_union_p) = (iou_rmse, psnr_intersection_rmse, psnr_union_rmse) if (
            prediction_rmse is not None) else (iou_p_iou, 0, 0)
    condition_coverage = HC if coverage_i > 0.2 else LC
    condition_intensity = HI if iou_i > 0.05 else LI
    condition = condition_coverage + condition_intensity
    
    # ----------------------------------------------------------------------------------
    # random result plot
    

    pl = path.split('/')
    filename = pl[-1].split('.')

    # if filename[0] in ['79', '126', '199', '729', '183', '992', '140', '189', '1159', '190', '26', '188']:
     # if filename[0] in ['78','240','249','0','6','19','2','10','14','15','27','807']:
    if 1:
        g1 = ImagePlot(GOES_UNITS,gf_max, gf_min,
                       input, 
                       GOES_input)
        # g2 = ImagePlot(GOES_UNITS,gf_max, gf_min,
        #                th_img_i,
        #                OTSU_thresholding_on_GOES + input_dis)
        g3 = ImagePlot(VIIRS_UNITS,vf_max,VIIRS_MIN_VAL,
                       groundTruth,
                       VIIRS_GROUND_TRUTH)
        g4 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else "IOU",vf_max,VIIRS_MIN_VAL,
                       prediction_rmse if prediction_rmse is not None else prediction_IOU,
                       Prediction_RMSE if prediction_rmse is not None else Prediction_JACCARD)
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
        logging.info(
        f'{LOSS_NAME},{condition},{filename[0]},{str(iou_rmse) if prediction_rmse is not None else ""},{str(iou_p_iou) if prediction_IOU is not None else ""},{psnr_intersection_rmse if prediction_rmse is not None else ""},{site_date_time}')
   
    return coverage_i, iou_i, psnr_intersection_i, psnr_union_i, iou_p, psnr_intersection_p, psnr_union_p, condition

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
            vmin,vmax = img_seq[col][row].vmin, img_seq[col][row].vmax
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
                if lable_blocks in [Prediction_JACCARD]:
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r", 2), vmin=vmin, vmax=vmax)
                else:
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
        path = '/'.join(pl[
                        :2]) + f"/{condition}/{filename[0]}.png" 
        # path = '/'.join(pl[
        #                 :2]) + f"/{condition}/{filename[0]}_{output_iou}_{psnr_intersection_i}_{psnr_union_i}_{str(round(coverage_i, 4))}.png"
        plt.rcParams['savefig.dpi'] = 600
        fig.savefig(path)
        # plt.show()
        plt.close()
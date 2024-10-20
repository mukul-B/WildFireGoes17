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
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from EvaluationReportingTemplate import EvaluatationReporting_reg
import wandb
from AutoEncoderTraining import balance_dataset_if_TH
from Autoencoder import Autoencoder, Encoder, Decoder
from CustomDataset import npDataset
from GlobalValues import COLOR_NORMAL_VALUE, RES_AUTOENCODER_PTH, RES_ENCODER_PTH, RES_DECODER_PTH, EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, GOES_Bands, model_path, \
    HC, HI, LI, LC
from GlobalValues import training_dir, Results, random_state, project_name_template, test_split, model_specific_postfix, realtime_model_specific_postfix
from ModelRunConfiguration import use_config
from EvaluationOperation import get_evaluation_results

plt.style.use('plot_style/wrf')

# logging.basicConfig(filename='evaluation.log', filemode='w',format='%(message)s', level=logging.INFO)

im_dir = training_dir
n_epochs = 1
batch_size = 1
random_seed = 1
torch.manual_seed(random_seed)
SMOOTH = 1e-6


def test(test_loader, selected_model, npd):
    avg_elapsed_time = 0.0
    count = 0.0
    dir = {}
    # Good Abstraction for reporting results
    evals = EvaluatationReporting_reg()
    # maxdif = 0
    iou_plot_control = []
    iou_plot_prediction = []
    # sum_fpr = []
    mc = []
    for i in range(0, 11):
        dir[i / 10] = (0, 0)
    # logging.info(dir)
    selected_model.eval()
    start_time = time.time()
    with torch.no_grad():
        # for batch_idx, (x, y) in enumerate(test_loader):
        for batch_idx, (x, y, z, gf_min, gf_max, vf_max) in enumerate(test_loader):
            count += 1
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            x = x.cuda()
            decoder_output = selected_model(x)
            if len(decoder_output) == 1:
                output_rmse, output_jaccard = None, None
                if LOSS_NAME == 'jaccard_loss':
                    output_jaccard = decoder_output
                else:
                    output_rmse = decoder_output
            else:
                output_rmse, output_jaccard = decoder_output[0], decoder_output[1]

            x = x.cpu()
            x = np.squeeze(x)
            y = np.squeeze(y)

            if output_rmse is not None:
                # output_rmse = output_rmse.view(1, 128, 128)
                output_rmse = output_rmse.cpu()
                output_rmse = np.squeeze(output_rmse)
            if output_jaccard is not None:
                output_jaccard = output_jaccard.cpu()
                output_jaccard = np.squeeze(output_jaccard)
            nonzero = np.count_nonzero(output_rmse)
            if True:
                path = f'{res}/{batch_idx}.png'
                gf_min, gf_max, vf_max = gf_min[0][0][0][0].item(), gf_max[0][0][0][0].item(), vf_max[0][0][0][0].item()
                eval_single = get_evaluation_results(output_rmse, output_jaccard, x, y, path, npd.array[batch_idx], gf_min, gf_max, vf_max,
                                 LOSS_NAME,z)
                evals.update(eval_single)
                iou_plot_control.append(eval_single.IOU_control)
                iou_plot_prediction.append(eval_single.IOU_predicted)
                mc.append(eval_single.coverage_converage)
                incv, incc = dir[round(eval_single.coverage_converage, 1)]
                dir[round(eval_single.coverage_converage, 1)] = (incv + eval_single.IOU_predicted, incc + 1)

    # logging.info(dir)
    for i in range(0, 11):
        incv, incc = dir[i / 10]
        dir[i / 10] = (0 if incc == 0 else (incv / incc), incc)
    # logging.info(dir)
    plot_result_histogram(count, iou_plot_prediction)
    evals.report_results()
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time
    logging.info("It took  {}s for processing  {} records".format(avg_elapsed_time, count))
    # print(f'{res}/evaluation.log')

def plot_result_histogram(count, iou_plot_prediction):
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 8))
    # hist_iou, bin_edges_iou = np.histogram(iou_plot_control, bins=100)
    # axs[0].hist(bin_edges_iou[:-1], bins=bin_edges_iou, weights=hist_iou / count)
    # axs[0].set_title('Distribution of control samples based on IOU', size=8)
    # axs[0].set_ylim([0, 1])
    # plt.scatter(sum_fpr,iou_plot_prediction)
    # axs.set_ylabel("IOU")
    # axs.set_xlabel('Total FRP of window')
    # axs.set_xlim([100,5000])
    # hist_iou_2, bin_edges_iou_2 = np.histogram(sum_fpr, bins=100)
    hist_iou_2, bin_edges_iou_2 = np.histogram(iou_plot_prediction, bins=100)
    axs.hist(bin_edges_iou_2[:-1], bins=bin_edges_iou_2, weights=hist_iou_2 / count)
    axs.set_title('Distribution of predictions  based on IOU', size=8)
    axs.set_ylim([0, 1])
    axs.set_ylabel("Fraction of samples")
    axs.set_xlabel('IOU')

    plt.show()

    fig.savefig(f'{res}/IOU_distribution.png')
    # plt.show()
    plt.close()

def test_runner(selected_model):

    # Get List of downloaded files and set up reference_data loader
    file_list = os.listdir(im_dir)
    file_list = balance_dataset_if_TH(file_list)
    logging.info(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=test_split, random_state=random_state)
    # test_files = os.listdir(im_dir)
    logging.info(f'{len(test_files)} test_data samples found')
    npd = npDataset(test_files, batch_size, im_dir, augment=False, evaluate=True)
    test_loader = DataLoader(npd)
    
    selected_model.cuda()

    # test the model components
    test(test_loader, selected_model, npd)

def get_selected_model_weight(selected_model,model_project_path):
    selected_model.load_state_dict(torch.load(model_project_path + "/" + RES_AUTOENCODER_PTH))


class RuntimeDLTransformation:
    def __init__(self,conf):
        loss_function = conf.get(LOSS_FUNCTION)
        loss_function_name = str(loss_function).split("'")[1].split(".")[1]
        
        self.LOSS_NAME = loss_function_name
        OUTPUT_ACTIVATION = loss_function(1).last_activation
        self.selected_model = Autoencoder(GOES_Bands, OUTPUT_ACTIVATION)
        model_name = type(self.selected_model).__name__

        project_name = project_name_template.format(
        model_name = model_name,
        loss_function_name=loss_function_name,
        n_epochs=conf.get(EPOCHS),
        batch_size=conf.get(BATCH_SIZE),
        learning_rate=conf.get(LEARNING_RATE),
        model_specific_postfix=realtime_model_specific_postfix
    )   
        print(project_name)
        path = model_path + project_name
        get_selected_model_weight(self.selected_model,path)
        self.selected_model.cuda()

    def Transform(self, x):
        
        x = single_dataload(x)
        with torch.no_grad():
            x = x.cuda()
            decoder_output = self.selected_model(x)
            if len(decoder_output) == 1:
                output_rmse, output_jaccard = None, None
                if self.LOSS_NAME == 'jaccard_loss':
                    output_jaccard = decoder_output
                else:
                    output_rmse = decoder_output
            else:
                output_rmse = decoder_output[0]
            return output_rmse
            
            
    def out_put_to_numpy(self, output_rmse):
        output_rmse = output_rmse.cpu()
        # x = x.cpu()
        # x = np.squeeze(x)
        output = np.squeeze(output_rmse)
        return output
        


def single_dataload(x):
    x = np.array(x) / float(COLOR_NORMAL_VALUE)
    # x = np.expand_dims(x, 1)
    x = torch.Tensor(x)
    return x


def prepare_dir(res):

    os.makedirs(Results, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    # if not os.path.exists(Results):
    #     os.mkdir(Results)
    # if not os.path.exists(res):
    #     os.mkdir(res)
    for coverage_type in [LC, HC]:
        for iou_type in [LI, HI]:
            type = coverage_type + iou_type
            if not os.path.exists(f'{res}/{type}'):
                os.mkdir(f'{res}/{type}')


def main(config=None):

    if config:
        wandb.config = config
    
    n_epochs=wandb.config.get(EPOCHS)
    batch_size=wandb.config.get(BATCH_SIZE)
    learning_rate=wandb.config.get(LEARNING_RATE)
    loss_function = wandb.config.get(LOSS_FUNCTION)

    loss_function_name = str(loss_function).split("'")[1].split(".")[1]

    global res, LOSS_NAME
    
    LOSS_NAME = loss_function_name
    OUTPUT_ACTIVATION = loss_function(1).last_activation
    
    selected_model = Autoencoder(GOES_Bands, OUTPUT_ACTIVATION)
    model_name = type(selected_model).__name__
    
    project_name = project_name_template.format(
    model_name = model_name,
    loss_function_name=loss_function_name,
    n_epochs=n_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    model_specific_postfix = model_specific_postfix
)
    path = model_path + project_name
    print(project_name)
    res = Results + project_name
    prepare_dir(res)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"{res}/evaluation.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(f'Evaluating Model : {project_name} at {path}')
    get_selected_model_weight(selected_model,path)
    test_runner(selected_model)
    print(f'{res}/evaluation.log')


if __name__ == "__main__":
    config = use_config
    main(config)

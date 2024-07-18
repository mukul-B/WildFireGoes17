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

import wandb
from Autoencoder import Encoder, Decoder
from AutoencoderDataset import npDataset
from GlobalValues import RES_ENCODER_PTH, RES_DECODER_PTH, EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, GOES_Bands, model_path, \
    HC, HI, LI, LC
from GlobalValues import training_dir, Results, random_state, project_name_template, test_split
from LossFunctionConfig import use_config
from EvaluationOperation import get_evaluation_results

plt.style.use('plot_style/wrf')

# logging.basicConfig(filename='evaluation.log', filemode='w',format='%(message)s', level=logging.INFO)

im_dir = training_dir
n_epochs = 1
batch_size = 1
random_seed = 1
torch.manual_seed(random_seed)
SMOOTH = 1e-6

class Evaluations:
    def __init__(self):
        self.evals = {
            'control': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'predicted': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'typecount': {}
        }

    def update(self, type, IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union):
        self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + IOU_control
        self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + psnr_control_inter
        self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + psnr_control_union
        self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + IOU_predicted
        self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + psnr_predicted_inter
        self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + psnr_predicted_union
        self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    def get_evaluations(self):
        return self.evals


def test(test_loader, encoder, decoder, npd):
    avg_elapsed_time = 0.0
    avg_IOU_control, avg_psnr_control_inter, avg_psnr_control_union, avg_IOU_predicted, avg_psnr_predicted_inter, avg_psnr_predicted_union = {}, {}, {}, {}, {}, {}
    typecount = {}
    count = 0.0
    dir = {}
    evals = Evaluations()
    # maxdif = 0
    iou_plot_control = []
    iou_plot_prediction = []
    # sum_fpr = []
    mc = []
    for i in range(0, 11):
        dir[i / 10] = (0, 0)
    # logging.info(dir)
    with torch.no_grad():
        # for batch_idx, (x, y) in enumerate(test_loader):
        for batch_idx, (x, y, z, gf_min, gf_max, vf_max) in enumerate(test_loader):
            count += 1
            # psnr_control = PSNR(x, y)
            # avg_psnr_control += psnr_control
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            # total_frp = torch.sum(z)
            # if (total_frp < 0):
            #     print("sgd")
            # sum_fpr.append(total_frp)
            x = x.cuda()
            start_time = time.time()
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time
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
                maxdif, IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union, type = \
                    get_evaluation_results(output_rmse, output_jaccard, x, y, path, npd.array[batch_idx], gf_min, gf_max, vf_max,
                                 LOSS_NAME)
                # avg_IOU_control[type] = avg_IOU_control.get(type, 0.0) + IOU_control
                # avg_psnr_control_inter[type] = avg_psnr_control_inter.get(type, 0.0) + psnr_control_inter
                # avg_psnr_control_union[type] = avg_psnr_control_union.get(type, 0.0) + psnr_control_union
                # avg_IOU_predicted[type] = avg_IOU_predicted.get(type, 0.0) + IOU_predicted
                # avg_psnr_predicted_inter[type] = avg_psnr_predicted_inter.get(type, 0.0) + psnr_predicted_inter
                # avg_psnr_predicted_union[type] = avg_psnr_predicted_union.get(type, 0.0) + psnr_predicted_union
                # typecount[type] = typecount.get(type, 0) + 1

                evals.update(type, IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union)

                iou_plot_control.append(IOU_control)
                iou_plot_prediction.append(IOU_predicted)
                mc.append(maxdif)
                incv, incc = dir[round(maxdif, 1)]
                dir[round(maxdif, 1)] = (incv + IOU_predicted, incc + 1)

    # logging.info(dir)
    for i in range(0, 11):
        incv, incc = dir[i / 10]
        dir[i / 10] = (0 if incc == 0 else (incv / incc), incc)
    # logging.info(dir)
    plot_result_histogram(count, iou_plot_prediction)
    # report_results(avg_IOU_control, avg_psnr_control_inter, avg_psnr_control_union, avg_IOU_predicted, avg_psnr_predicted_inter, avg_psnr_predicted_union, typecount, count)
    report_results(evals.get_evaluations())
    logging.info("It took  {}s for processing  {} records".format(avg_elapsed_time, count))

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

def report_results(evaluations):

    st = ''.join(['-' for _ in range(128)])
    logging.info(st)
    logging.info("| {:<25}| {:<20}| {:<20}| {:<20}| {:<20}| {:<10}|".format("Type", "Control/Predicted", "IOU",
                                                                            "PSNR_intersection", "PSNR_union", "Count"))
    logging.info(st)
    count = sum(evaluations['typecount'].values())
    for type in evaluations['typecount']:
    # for coverage_type in [LC, HC]:
    #     for iou_type in [LI, HI]:
            # type = coverage_type + iou_type
            # type_name = coverage_type + iou_type
            # if type not in evaluations['typecount']:
            #     continue
            type_name = type
            count2 = evaluations['typecount'][type]
            
            avg_IOU_predicted = evaluations['predicted']['avg_IOU']
            avg_IOU_control = evaluations['control']['avg_IOU']
            avg_psnr_predicted_inter = evaluations['predicted']['avg_psnr_inter']
            avg_psnr_control_inter = evaluations['control']['avg_psnr_inter']
            avg_psnr_predicted_union = evaluations['predicted']['avg_psnr_union']
            avg_psnr_control_union = evaluations['control']['avg_psnr_union']

            IOU_predicted = avg_IOU_predicted[type] / count2
            IOU_control = avg_IOU_control[type] / count2
            PSNR_predicted_intersection = avg_psnr_predicted_inter[type] / count2
            PSNR_control_intersection = avg_psnr_control_inter[type] / count2
            PSNR_predicted_union = avg_psnr_predicted_union[type] / count2
            PSNR_control_union = avg_psnr_control_union[type] / count2

            logging.info("| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Control", IOU_control,
                                                                                     PSNR_control_intersection,
                                                                                     PSNR_control_union, count2))
            logging.info(
                "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Predicted", IOU_predicted,
                                                                            PSNR_predicted_intersection,
                                                                            PSNR_predicted_union, count2))

    logging.info(st)

    logging.info(
        "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Control",
                                                                    sum(avg_IOU_control.values()) / count,
                                                                    sum(avg_psnr_control_inter.values()) / count,
                                                                    sum(avg_psnr_control_union.values()) / count,
                                                                    count))
    logging.info(
        "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Predicted",
                                                                    sum(avg_IOU_predicted.values()) / count,
                                                                    sum(avg_psnr_predicted_inter.values()) / count,
                                                                    sum(avg_psnr_predicted_union.values()) / count,
                                                                    count))
    logging.info(st)

def test_runner(npd):
    test_loader = DataLoader(npd)

    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(GOES_Bands)
    decoder = Decoder(256, OUTPUT_ACTIVATION)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.cuda()
    decoder.cuda()

    # test the model components
    test(test_loader, encoder, decoder, npd)


def supr_resolution(conf, x):
    loss_function = conf.get(LOSS_FUNCTION)
    loss_function_name = str(loss_function).split("'")[1].split(".")[1]
    project_name = project_name_template.format(
    loss_function_name=loss_function_name,
    n_epochs=conf.get(EPOCHS),
    batch_size=conf.get(BATCH_SIZE),
    learning_rate=conf.get(LEARNING_RATE)
)
    # project_name = f"wildfire_{loss_function_name}_{conf.get(EPOCHS)}epochs_{conf.get(BATCH_SIZE)}batchsize_{conf.get(LEARNING_RATE)}lr"
    path = model_path + project_name
    LOSS_NAME = loss_function_name
    OUTPUT_ACTIVATION = loss_function(1).last_activation
    encoder_path = path + "/" + RES_ENCODER_PTH
    decoder_path = path + "/" + RES_DECODER_PTH
    encoder = Encoder(GOES_Bands)
    decoder = Decoder(256, OUTPUT_ACTIVATION)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.cuda()
    decoder.cuda()

    x = np.array(x) / 255.
    x = np.expand_dims(x, 1)
    x = torch.Tensor(x)
    with torch.no_grad():
        x = x.cuda()
        encoder_output = encoder(x)
        decoder_output = decoder(encoder_output)
        if len(decoder_output) == 1:
            output_rmse, output_jaccard = None, None
            if LOSS_NAME == 'jaccard_loss':
                output_jaccard = decoder_output
            else:
                output_rmse = decoder_output
        else:
            output_rmse = decoder_output[0]
        output_rmse = output_rmse.cpu()
        x = x.cpu()
        x = np.squeeze(x)
        output = np.squeeze(output_rmse)
        return output


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
    # Get List of downloaded files and set up reference_data loader
    if config:
        wandb.config = config

    loss_function = wandb.config.get(LOSS_FUNCTION)
    loss_function_name = str(loss_function).split("'")[1].split(".")[1]
    # config.py

    
    project_name = project_name_template.format(
    loss_function_name=loss_function_name,
    n_epochs=wandb.config.get(EPOCHS),
    batch_size=wandb.config.get(BATCH_SIZE),
    learning_rate=wandb.config.get(LEARNING_RATE)
)
    # project_name = f"wildfire_{loss_function_name}_{wandb.config.get(EPOCHS)}epochs_{wandb.config.get(BATCH_SIZE)}batchsize_{wandb.config.get(LEARNING_RATE)}lr"
    print(project_name)

    global encoder_path, decoder_path, res, OUTPUT_ACTIVATION, LOSS_NAME
    path = model_path + project_name
    LOSS_NAME = loss_function_name
    # logging.info(f'LOSS_NAME : {LOSS_NAME}')
    # path = "Model_2b5S1rm"
    encoder_path = path + "/" + RES_ENCODER_PTH
    decoder_path = path + "/" + RES_DECODER_PTH
    OUTPUT_ACTIVATION = loss_function(1).last_activation
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
    file_list = os.listdir(im_dir)
    logging.info(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=test_split, random_state=random_state)
    # test_files = os.listdir(im_dir)
    logging.info(f'{len(test_files)} test_data samples found')
    npd = npDataset(test_files, batch_size, im_dir, augment=False, evaluate=True)
    test_runner(npd)
    print(f'{res}/evaluation.log')


if __name__ == "__main__":
    config = use_config
    main(config)

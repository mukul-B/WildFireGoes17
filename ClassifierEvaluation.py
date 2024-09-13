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
from Classifier import Encoder
from CustomDataset import npDataset
from ClassifierTraining import balance_dataset_if_TH
from GlobalValues import RES_ENCODER_PTH, RES_DECODER_PTH, EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, GOES_Bands, model_path, \
    HC, HI, LI, LC
from GlobalValues import training_dir, Results, random_state, project_name_template, test_split, model_specific_postfix
from ModelRunConfiguration import use_config
from EvaluationOperation import  get_evaluation_results
from LossFunctions import Classification_loss
from TransferLearning import get_pre_model

plt.style.use('plot_style/wrf')

# logging.basicConfig(filename='evaluation.log', filemode='w',format='%(message)s', level=logging.INFO)

im_dir = training_dir
n_epochs = 1
batch_size = 1
random_seed = 1
torch.manual_seed(random_seed)
SMOOTH = 1e-6

class EvaluateDataset:
    def __init__(self):
        self.evals = {
            'control': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {},
                'avg_acuracy': {},
                'avg_firesize': {},
                'avg_FRP': {},
                'max_firesize': {},
                'max_FRP': {},
                'min_firesize': {},
                'min_FRP': {}
            },
            'predicted': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {},
                'avg_acuracy': {},
                'avg_firesize': {},
                'avg_FRP': {},
                'max_firesize': {},
                'max_FRP': {},
                'min_firesize': {},
                'min_FRP': {}
            },
            'typecount': {}
        }

    # def update(self, type, IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union):
    #     eval_single.type, eval_single.IOU_control, eval_single.psnr_control_inter, eval_single.psnr_control_union, eval_single.IOU_predicted, eval_single.psnr_predicted_inter, eval_single.psnr_predicted_union
    #     self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + IOU_control
    #     self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + psnr_control_inter
    #     self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + psnr_control_union
    #     self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + IOU_predicted
    #     self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + psnr_predicted_inter
    #     self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + psnr_predicted_union
    #     self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    def undate_single_value(self, index, value,type):
        index[type] = index.get(type, 0.0) + value

    def update(self, eval_single):
        
        type = eval_single.type
        self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + eval_single.IOU_control
        self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_control_inter
        self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_control_union
        self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + eval_single.IOU_predicted
        self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_predicted_inter
        self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_predicted_union
        self.evals['predicted']['avg_acuracy'][type] = self.evals['predicted']['avg_acuracy'].get(type, 0.0) + eval_single.coverage_converage
        # self.undate_single_value(self.evals['predicted']['avg_IOU'],eval_single.IOU_control,type)
        self.evals['predicted']['avg_firesize'][type] = self.evals['predicted']['avg_firesize'].get(type, 0.0) + eval_single.fire_size
        self.evals['predicted']['avg_FRP'][type] = self.evals['predicted']['avg_FRP'].get(type, 0.0) + eval_single.total_FRP
        self.evals['predicted']['max_firesize'][type] = max(self.evals['predicted']['max_firesize'].get(type, 0.0) , eval_single.fire_size)
        self.evals['predicted']['max_FRP'][type] = max(self.evals['predicted']['max_FRP'].get(type, 0.0) , eval_single.total_FRP)
        self.evals['predicted']['min_firesize'][type] = min(self.evals['predicted']['min_firesize'].get(type, 128*128) , eval_single.fire_size)
        self.evals['predicted']['min_FRP'][type] = min(self.evals['predicted']['min_FRP'].get(type, 128*128*400) , eval_single.total_FRP)
        
        self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1
        if(type == 'HCHI' and self.evals['predicted']['min_firesize'][type]  == 0):
            print("hpw")

    def get_evaluations(self):
        return self.evals
    
    def report_results(self):
        # Get the dynamic metric keys from the first category (e.g., 'control')
        metric_keys = list(self.evals['control'].keys())
        
        # Build the header line based on metric keys
        header_line = '-' * (26 + 22 * len(metric_keys) + 12)
        logging.info(header_line)
        logging.info("| {:<25}| {:<20}|".format("Type", "Control/Predicted") + " | ".join([f"{key:<20}" for key in metric_keys]) + "| {:<10}|".format("Count"))
        logging.info(header_line)
        
        for type_name in self.evals['typecount']:
            typecount = self.evals['typecount'][type_name]
            self._log_type_evaluation(type_name, "control", typecount, metric_keys)
            self._log_type_evaluation(type_name, "predicted", typecount, metric_keys)

        logging.info(header_line)
        self._log_total_evaluation("control", metric_keys)
        self._log_total_evaluation("predicted", metric_keys)
        logging.info(header_line)

        avg_IOU_predicted = self.evals['predicted']['avg_IOU']
        avg_psnr_predicted_inter = self.evals['predicted']['avg_psnr_inter']
        avg_psnr_predicted_union = self.evals['predicted']['avg_psnr_union']
        coverage = self.evals['predicted']['avg_acuracy']

        TP = sum(avg_IOU_predicted.values())
        FN = sum(avg_psnr_predicted_inter.values())
        FP = sum(avg_psnr_predicted_union.values())
        TN = sum(coverage.values())

        # Calculate accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        logging.info(f"Accuracy ( how many prediction were right): {accuracy}" )

        # Calculate precision
        precision = TP / (TP + FP)
        logging.info(f"Precision (how many times fire prediction is right): {precision}" )

        # Calculate recall
        recall = TP / (TP + FN)
        logging.info(f"Recall (how many wildfire were predicted): {recall}")

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)

        logging.info(f"F1 Score: {f1}")

    def _log_type_evaluation(self, type_name, category, typecount, metric_keys):
        values = [self.evals[category][key].get(type_name , 0.0) / (typecount if key.startswith('avg') else 1)  for key in metric_keys]

        logging.info("| {:<25}| {:<20}|".format(type_name, category) + " | ".join([f"{value:<20}" for value in values]) + "| {:<10}|".format(typecount))

    def _log_total_evaluation(self, category, metric_keys):

        
    
        count = sum(self.evals['typecount'].values())
        avg_values = [sum(self.evals[category][key].values()) / count for key in metric_keys]

        logging.info("| {:<25}| {:<20}|".format("Total", category) + " | ".join([f"{value:<20}" for value in avg_values]) + "| {:<10}|".format(count))


def test(test_loader, selected_model, npd):
    avg_elapsed_time = 0.0
    count = 0.0
    dir = {}
    evals = EvaluateDataset()
    # maxdif = 0
    iou_plot_control = []
    iou_plot_prediction = []

    iou_plot_control2 = []
    iou_plot_prediction2 = []
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
            _, decoder_output = torch.max(decoder_output, 1)
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
                # print(z.nonzero().size(0), z.sum().item())
                eval_single.fire_size = y.nonzero().size(0)
                eval_single.total_BT = y.sum().item()
                eval_single.total_FRP = z.sum().item()
                eval_single.G7_max = round(x[0].max().item(),4)
                eval_single.G14_max = round(x[1].max().item(),4)
                
                # if (eval_single.fire_size < 20 and eval_single.total_FRP < 50 ):
                evals.update(eval_single)
                if(eval_single.type == 'HCHI'):
                    iou_plot_control.append([eval_single.IOU_predicted,eval_single.fire_size * (100/16384),eval_single.total_FRP,eval_single.total_BT,eval_single.G7_max,eval_single.G14_max ])
                if(eval_single.type == 'LCLI'):
                    iou_plot_prediction.append([eval_single.IOU_predicted,eval_single.fire_size * (100/16384),eval_single.total_FRP,eval_single.total_BT,eval_single.G7_max,eval_single.G14_max  ])

                if(eval_single.type == 'HCLI'):
                    iou_plot_control2.append([eval_single.IOU_predicted,eval_single.fire_size * (100/16384),eval_single.total_FRP,eval_single.total_BT,eval_single.G7_max,eval_single.G14_max  ])
                if(eval_single.type == 'LCHI'):
                    iou_plot_prediction2.append([eval_single.IOU_predicted,eval_single.fire_size * (100/16384),eval_single.total_FRP,eval_single.total_BT,eval_single.G7_max,eval_single.G14_max  ])
                mc.append(eval_single.coverage_converage)
                incv, incc = dir[round(eval_single.coverage_converage, 1)]
                dir[round(eval_single.coverage_converage, 1)] = (incv + eval_single.IOU_predicted, incc + 1)

    # logging.info(dir)
    for i in range(0, 11):
        incv, incc = dir[i / 10]
        dir[i / 10] = (0 if incc == 0 else (incv / incc), incc)
    # logging.info(dir)

    # plot_result_histogram(count, iou_plot_prediction)
    evals.report_results()
    import pandas as pd 
    df = pd.DataFrame(iou_plot_control)
    df.to_csv("HCHI.csv")
    df2 = pd.DataFrame(iou_plot_prediction)
    df2.to_csv("LCLI.csv")
    df = pd.DataFrame(iou_plot_control)
    df.to_csv("HCLI.csv")
    df2 = pd.DataFrame(iou_plot_prediction)
    df2.to_csv("LCHI.csv")
    plot_classification_dependency(iou_plot_control, iou_plot_prediction,f'{res}/IOU_foreachcoverage.png')
    plot_classification_dependency(iou_plot_control, [],f'{res}/IOU_foreachcoverage2.png')
    plot_classification_dependency(iou_plot_control2, iou_plot_prediction2,f'{res}/IOU_foreachcoverage3.png')
    
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time
    logging.info("It took  {}s for processing  {} records".format(avg_elapsed_time, count))

def plot_classification_dependency(iou_plot_control, iou_plot_prediction,savePath):
    iou_plot_control = np.array(iou_plot_control)
    iou_plot_prediction = np.array(iou_plot_prediction)

    if(len(iou_plot_control) > 0):
        plt.scatter(iou_plot_control[:, 1], iou_plot_control[:, 2], color='green', s=10, label='True Positive')
    if(len(iou_plot_prediction) > 0):
        plt.scatter(iou_plot_prediction[:, 1], iou_plot_prediction[:, 2], color='red', alpha=0.5, s=10, label='False Negative')

    # Add labels and title
    plt.xlabel('Fire Size ( % in 128 x 128 pixel frame)')
    plt.ylabel('Total FRP')
    plt.title('FRP and FIRE SIZE influence on classification')

    plt.legend(loc='upper right')  
    plt.tight_layout()

    plt.savefig(savePath)
    plt.close()
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


    train_files, test_files = train_test_split(file_list, test_size=test_split, random_state=random_state)
    # # train_files, validation_files = train_test_split(train_files, test_size=validation_split, random_state=random_state)

    # train_pos, test_pos = train_test_split(file_list_pos, test_size=test_split, random_state=random_state) if(len(file_list_pos)>0) else [[],[]]
    # train_neg, test_neg = train_test_split(file_list_neg, test_size=test_split, random_state=random_state) if(len(file_list_neg)>0) else [[],[]]
    # train_TH, test_TH = train_test_split(file_list_TH, test_size=test_split, random_state=random_state) if(len(file_list_TH)>0) else [[],[]]

    # # Combine the splits
    # train_files = train_pos + train_neg + train_TH
    # test_files = test_pos + test_neg + test_TH
    logging.info(
        f'Training sample : {len(train_files)} , testing samples : {len(test_files)}')
    npd = npDataset(test_files, batch_size, im_dir, augment=False, evaluate=True)
    test_loader = DataLoader(npd)
    
    selected_model.cuda()

    # test the model components
    test(test_loader, selected_model, npd)

def get_selected_model_weight(selected_model,model_project_path):
    # selected_model.load_state_dict(torch.load(model_project_path + "/" + RES_AUTOENCODER_PTH))
    selected_model.load_state_dict(torch.load(model_project_path + "/" + RES_ENCODER_PTH))

    # selected_model.decoder.load_state_dict(torch.load(model_project_path + "/" + RES_DECODER_PTH))


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

    selected_model = get_pre_model(GOES_Bands)
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

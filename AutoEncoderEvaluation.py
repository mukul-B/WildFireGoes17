# -*- coding: utf-8 -*-
"""
This script will run through the directory of training images, load
the image pairs, and then batch them before feading them into a pytorch based
autoencoder using MSE reconstruction loss for the superresolution.

Created on Sun Jul 26 11:17:09 2020

@author: mullenj
@modified by: mukul badhan
on Sun Jul 23 11:17:09 2022
"""
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Autoencoder import Encoder, Decoder
from AutoencoderDataset import npDataset
from GlobalValues import testing_dir, training_dir, Results, model_path

im_dir2 = testing_dir
im_dir = training_dir

encoder_path = model_path+'SuperRes_Encoder.pth'
decoder_path = model_path+'SuperRes_Decoder.pth'
n_epochs = 1
batch_size = 1
random_seed = 1
torch.manual_seed(random_seed)


def test(test_loader, encoder, decoder):
    avg_psnr_predicted = 0.0
    avg_psnr_control = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            count += 1
            print("Processing ", batch_idx)
            psnr_bicubic = PSNR(x, y)
            avg_psnr_control += psnr_bicubic
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            x = x.cuda()
            start_time = time.time()
            encoder_output = encoder(x)
            output = decoder(encoder_output)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            output = output.cpu()
            x = x.cpu()

            psnr_predicted = PSNR(output, y)
            avg_psnr_predicted += psnr_predicted

            x = np.squeeze(x)
            output = np.squeeze(output)
            y = np.squeeze(y)

            nonzero = np.count_nonzero(y)
            # print(nonzero,psnr_predicted)
            # print(output.max())
            # if True:
            if nonzero > 20 and psnr_predicted > 60:
                # if count % 100 == 0:
                fig, axs = plt.subplots(3, 1, constrained_layout=True)
                axs[0].imshow(x)
                axs[0].set_title('GOES Imagery')
                axs[2].imshow(output)
                # print(output)
                axs[2].set_title('Network Prediction')
                axs[1].imshow(y)
                axs[1].set_title('VIIRS-I Imagery')
                fig.savefig(f'{Results}/{batch_idx}_{nonzero}_{psnr_predicted}.png')
                plt.close()

    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_control=", avg_psnr_control / count)
    print("It takes average {}s for processing".format(avg_elapsed_time / count))


def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    imdff = imdff.flatten()
    rmse = math.sqrt(np.mean(np.array(imdff ** 2)))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def test_runner(npd):
    test_loader = DataLoader(npd)

    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(1)
    decoder = Decoder(256)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.cuda()
    decoder.cuda()

    # test the model components
    test(test_loader, encoder, decoder)


def supr_resolution(x):
    encoder = Encoder(1)
    decoder = Decoder(256)
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
        output = decoder(encoder_output)
        output = output.cpu()
        x = x.cpu()
        x = np.squeeze(x)
        output = np.squeeze(output)
        return output


def prepare_dir():
    if not os.path.exists(Results):
        os.mkdir(Results)


def main():
    # Get List of downloaded files and set up reference_data loader
    prepare_dir()
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    # test_files = os.listdir(im_dir)
    print(f'{len(test_files)} reference_data samples found')
    npd = npDataset(test_files, batch_size, im_dir)
    test_runner(npd)


if __name__ == "__main__":
    main()

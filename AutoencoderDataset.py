import os
from random import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from GlobalValues import GOES_Bands, training_data_field_names, gf_c_fields, COLOR_NORMAL_VALUE


class npDataset(Dataset):
    """
    npDataset will take in the list of numpy files and create a torch dataset
    out of it.
    """

    def __init__(self, data_list, batch_size, im_dir,augment,evaluate):
        self.array = data_list
        self.batch_size = batch_size
        self.im_dir = im_dir
        self.transform = transform
        self.augment = augment
        self.evaluate = evaluate

    def __len__(self): return int((len(self.array) / self.batch_size))

    def __getitem__(self, i):
        """
        getitem will first select the batch of files before loading the files
        and splitting them into the goes and viirs components, the input and target
        of the network
        """
        files = self.array[i * self.batch_size:i * self.batch_size + self.batch_size]
        x = []
        y = []
        if self.evaluate:
            z = []
            a = []
            b = []
            c = []
        for file in files:
            file_path = os.path.join(self.im_dir, file)
            sample = np.load(file_path)
            training_data = {name: sample[:, :, i] for i, name in enumerate(training_data_field_names)}
            x.append(np.stack([training_data[field] for field in gf_c_fields], axis=0))
            y.append(training_data['vf'])
            if (self.evaluate):
                z.append(training_data['vf_FRP'])
                a.append(training_data['gf_min'])
                b.append(training_data['gf_max'])
                c.append(training_data['vf_max'])
        x, y = np.array(x) / float(COLOR_NORMAL_VALUE), np.array(y) / float(COLOR_NORMAL_VALUE)
        y =  np.expand_dims(y, 1)
        # print(i,x.sum(),y.sum())
        # x, y = np.expand_dims(x, 1) if GOES_Bands == 1 else x, np.expand_dims(y, 1)
        x,y = torch.Tensor(x), torch.Tensor(y)
        if(self.augment):
            # TODO not doing transformation increases the accuracy in testing (need to be checked)
            x, y = self.transform(x,y)
        if self.evaluate:
            z = np.array(z)
            z = np.expand_dims(z, 1)
            z = torch.Tensor(z)
            return x,y,z,a,b,c
        return x,y



# transform = transforms.Compose([
#     # transforms.ToPILImage(),
#     # transforms.ToTensor(),
#     transforms.RandomHorizontalFlip()
#
# ])


def transform( image, mask):
    # # Resize
    # resize = transforms.Resize(size=(520, 520))
    # image = resize(image)
    # mask = resize(mask)
    #
    # # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=(512, 512))
    # image = torch.crop(image, i, j, h, w)
    # mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random() > 0.5:
        image = image.flip((2,))
        mask = mask.flip((2,))

    # Random vertical flipping
    if random() > 0.5:
        image = image.flip( (3,))
        mask = mask.flip( (3,))

    return image, mask

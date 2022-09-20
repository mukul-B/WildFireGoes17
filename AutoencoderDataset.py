import os

import numpy as np
import torch
from torch.utils.data import Dataset


class npDataset(Dataset):
    """
    npDataset will take in the list of numpy files and create a torch dataset
    out of it.
    """

    def __init__(self, data_list, batch_size,im_dir):
        self.array = data_list
        self.batch_size = batch_size
        self.im_dir = im_dir

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
        for file in files:
            file_path = os.path.join(self.im_dir, file)
            sample = np.load(file_path)
            x.append(sample[:, :, 1])
            y.append(sample[:, :, 0])
        x, y = np.array(x) / 255., np.array(y) / 255.
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        return torch.Tensor(x), torch.Tensor(y)

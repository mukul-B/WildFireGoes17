from torchvision import models
import torch.nn as nn


def get_pre_model(num_fltrs):
    model = models.resnet34(pretrained= True)
    num_fltrs = model.fc.in_features
    model.fc = nn.Linear(num_fltrs,2)
    return model
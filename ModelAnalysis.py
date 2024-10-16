
from Classifier import Encoder
from torchinfo import summary

from Unet import UNET
from Autoencoder import Autoencoder

model = Autoencoder(in_features=3, outtype='relu')

# model = UNET(in_channels=3, out_channels=1)
summary(model, input_size=(1,3, 128, 128))
print("----------------------------------------------------")
print(model)
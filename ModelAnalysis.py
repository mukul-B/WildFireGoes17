
from Classifier import Encoder
from torchinfo import summary


model = Encoder(in_features=3)
summary(model, input_size=(16,3, 128, 128))
print("----------------------------------------------------")
print(model)
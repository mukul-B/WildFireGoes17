import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Autoencoder trained to separate class distributions in the latent space
    This is the encoder portion
    """
    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.in_features = in_features

        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        # self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        # self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        # self.bn6 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(256 * 16 * 16, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=(2, 2))
        
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=(2, 2))
        
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = F.max_pool2d(F.relu(self.bn6(self.conv6(x))), kernel_size=(2, 2))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=(2, 2))
        
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=(2, 2))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# Example usage:
# model = Encoder(in_features=3)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


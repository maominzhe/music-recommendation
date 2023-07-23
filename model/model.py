import torch
import torch.nn as nn
import torch.nn.functional as F


# AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Convolution Input channel 1, output channel 64 Convolution kernel size 11*11 Step size 4 Zero padding 2
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            # ReLU
            nn.ReLU(inplace=True),
            # Maxpooling
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            # FUlly connected layers
            #nn.Linear(12288, 1024),
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            # Dropout 
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        #x = x.view(-1, 3072)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
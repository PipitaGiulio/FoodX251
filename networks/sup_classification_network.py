import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

### Definitive Supervised Learning Classification Network Architecture
class SupClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        #224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        #112
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #56
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #28
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        #14
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.adpool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Linear(256 * 9, 1024),
            nn.Dropout(p = 0.3),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p = 0.3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 251)
        )
    
    def forward(self, x):
        c1 = self.conv1(x)
        x = self.maxpool(c1)
        c2 = self.conv2(x)
        x = self.maxpool(c2)
        c3 = self.conv3(x)
        x = self.maxpool(c3)
        c4 = self.conv4(x)
        x = self.maxpool(c4)
        c5 = self.conv5(x)
        x = self.adpool(c5)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x
 



### First attempt at the creation of a Supervised Learning Classification Network
class SupClassificationNetworkOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        #224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #112
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #56
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        #28
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        #14
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.adpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear( 512, 251)
        )
    
    def forward(self, x):
        c1 = self.conv1(x)
        x = self.maxpool(c1)
        c2 = self.conv2(x)
        x = self.maxpool(c2)
        c3 = self.conv3(x)
        x = self.maxpool(c3)
        c4 = self.conv4(x)
        x = self.maxpool(c4)
        c5 = self.conv5(x)
        x = self.adpool(c5)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x
 
    
if __name__ == '__main__':
    net = SupClassificationNetwork()
    summary(net, input_size=(3, 256, 256))
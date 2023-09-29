# customModels.py
# this file holds some custom models created for this TRC project
import torch.nn as nn
import torch
import torch.optim as optim

class ProjectNN3(nn.Module):
    def __init__(self):
        super(ProjectNN3, self).__init__()

        # define the layers of the neural network
        self.features = nn.Sequential(
            #slightly modified version of alexnet to make it smaller

            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(5408, 100),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        return out

class ProjectNN2(nn.Module):
    def __init__(self):
        super(ProjectNN2, self).__init__()

        # define the layers of the neural network
        self.features = nn.Sequential(
            #slightly modified version of alexnet to make it smaller

            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(5408, 100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        return out

class ProjectNN(nn.Module):
    def __init__(self):
        super(ProjectNN, self).__init__()

        # define the layers of the neural network
        self.features = nn.Sequential(
            #slightly modified version of alexnet to make it smaller

            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2028, 100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        return out

# this is the default class that i used. I am not changing the name for fear of breaking old code other network models will be based off this one
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(NeuralNetwork, self).__init__()

        # define the layers of the neural network
        self.features = nn.Sequential(
            #slightly modified version of alexnet to make it smaller

            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(3, 64, (3, 3), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # this one i need to recalculate
            nn.Linear(6912, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            # it seems like i dont need a softmax activation function here, as it automatically does it with cross entropy loss
        )

        print('define the self')

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        return out


class BlackAndWhiteNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super(BlackAndWhiteNN, self).__init__()
        # the only modification here is that the first layer takes in 1 channel input instead of 3 channel input
        # define the layers of the neural network
        self.features = nn.Sequential(
            #slightly modified version of alexnet to make it smaller

            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(1, 64, (3, 3), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # this one i need to recalculate
            nn.Linear(6912, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            # it seems like i dont need a softmax activation function here, as it automatically does it with cross entropy loss
        )

        print('define the self')

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)   # this one is for the batches, to resize it so that it wont have an issue
        out = self.classifier(out)
        return out
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Linear, Dropout
import torch
from torchinfo import summary

# this is adapted from https://github.com/Moeo3/GoogLeNet-Inception-V3-pytorch/blob/master/googlenet_v3.py#L58 with its size modified to suit the CIFAR10 dataset instead of the origianl ImageNet dataset.

''' comments on models
the orignal model has
3 conv_bn
1 pool
2 conv_bn
1 pool
3x inception a
1x inception b
4x inception c
1x inception d
2x inception e
1 conv_bn
1 adaptive pool 2d

dropout
flatten

fully connected layer

===================================
for our model, we are gonna just makeit smaller so that it trains faster, also cifar10 does not have the 1000 classes in imagenet lol
'''

class InceptionModel1(Module):
    def __init__(self, channels_in, class_num = 10):
        super(InceptionModel1, self).__init__()
        # remember, i must be able to extract the feature maps of each of the convolution layers, and as such, i must design my network around that as well

        # if this one is false, it will return feature maps. This would be found at the return funtion in the forward function
        self.PCA = False
        # input is N, 3, 32, 32
        self.layer1 = Sequential(
            Conv2d_BN(channels_in = channels_in, channels_out= 32, kernel_size=3, stride=2, padding=1),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)
        )
        # N, 32, 16, 16
        self.layer2 = Sequential(
            Conv2d_BN(channels_in = 32, channels_out= 64, kernel_size=3, stride=2, padding=1),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)
        )
        # N, 64, 8, 8

        # going into the inception layers
        # note that each of the components inside inception will ALWAYS retain the same width and height, and all the channels are concatenated together thats all
        self.incep1 = InceptionA(64, 16)
        # N, 240, 8, 8
        self.incep2 = InceptionB(240)
        # N, 720, 8, 8
        # inception 3 barely fits CIFAR10, i think this will be the last layer
        self.incep3 = InceptionC(720, 128)
        # N, 768, 8, 8
        self.incep4 = InceptionD(768)
        # N, 1280, 8, 8
        self.incep5 = InceptionE(1280)
        # N, 2048, 8, 8

        # going into the output layer now, last conv layer and then flattening it
        self.out = Sequential(
            # lowering the number of channels
            Conv2d_BN(channels_in = 2048, channels_out= 1024, kernel_size=1, stride=1, padding=0),
            AdaptiveAvgPool2d(1),
            Dropout(0.5)
        )
        # N, 1024, 1, 1

        # this one will then output it based on the number of classes, based on softmax i guess at this point
        self.fc = Linear(1024, class_num)

    def forward(self, x):
        x = self.layer1(x)
        fmap1 = x.clone()
        x = self.layer2(x)
        fmap2 = x.clone()
        x = self.incep1(x)
        fmap3 = x.clone()
        x = self.incep2(x)
        fmap4 = x.clone()
        x = self.incep3(x)
        fmap5 = x.clone()
        x = self.incep4(x)
        fmap6 = x.clone()
        x = self.incep5(x)
        fmap7 = x.clone()
        x = self.out(x)
        fmap8 = x.clone()
        # this one is to flatten, and retaining the batch size
        x = torch.flatten(x, 1)
        x = self.fc(x)
        fmap9 = x.clone()

        if self.PCA:
            return x, (fmap1, fmap2, fmap3, fmap3, fmap4, fmap5, fmap6, fmap7, fmap8, fmap9)
        else:
            return x

class InceptionModel2(Module):
    def __init__(self, channels_in, class_num = 10):
        super(InceptionModel2, self).__init__()
        # remember, i must be able to extract the feature maps of each of the convolution layers, and as such, i must design my network around that as well

        # if this one is false, it will return feature maps. This would be found at the return funtion in the forward function
        self.PCA = False
        # input is N, 3, 32, 32
        self.layer1 = Sequential(
            Conv2d_BN(channels_in = channels_in, channels_out= 32, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 32, 16, 16
        self.layer2 = Sequential(
            Conv2d_BN(channels_in = 32, channels_out= 64, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 64, 8, 8

        # going into the inception layers
        # note that each of the components inside inception will ALWAYS retain the same width and height, and all the channels are concatenated together thats all
        self.incep1 = InceptionA(64, 16)
        # N, 240, 8, 8
        self.incep2 = InceptionB(240)
        # N, 720, 8, 8
        # inception 3 barely fits CIFAR10, i think this will be the last layer
        self.incep3 = InceptionC(720, 128)
        # N, 768, 8, 8

        # going into the output layer now, last conv layer and then flattening it
        self.out = Sequential(
            # lowering the number of channels
            Conv2d_BN(channels_in = 768, channels_out= 320, kernel_size=1, stride=1, padding=0),
            AdaptiveAvgPool2d(1),
            Dropout(0.5)
        )
        # N, 1024, 1, 1

        # this one will then output it based on the number of classes, based on softmax i guess at this point
        self.fc = Linear(320, class_num)

    def forward(self, x):
        x = self.layer1(x)
        fmap1 = x.clone()
        x = self.layer2(x)
        fmap2 = x.clone()
        x = self.incep1(x)
        fmap3 = x.clone()
        x = self.incep2(x)
        fmap4 = x.clone()
        x = self.incep3(x)
        fmap5 = x.clone()
        x = self.out(x)
        fmap8 = x.clone()
        # this one is to flatten, and retaining the batch size
        x = torch.flatten(x, 1)
        x = self.fc(x)
        fmap9 = x.clone()

        if self.PCA:
            return x, (fmap1, fmap2, fmap3, fmap3, fmap4, fmap5, fmap8, fmap9)
        else:
            return x

class InceptionModel3(Module):
    def __init__(self, channels_in, class_num = 10):
        super(InceptionModel3, self).__init__()
        # remember, i must be able to extract the feature maps of each of the convolution layers, and as such, i must design my network around that as well

        # if this one is false, it will return feature maps. This would be found at the return funtion in the forward function
        self.PCA = False
        # input is N, 3, 32, 32
        self.layer1 = Sequential(
            Conv2d_BN(channels_in = channels_in, channels_out= 32, kernel_size=3, stride=1, padding=1)
            #MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 32, 16, 16
        self.layer2 = Sequential(
            Conv2d_BN(channels_in = 32, channels_out= 64, kernel_size=3, stride=1, padding=1)
            #MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 64, 8, 8

        # going into the inception layers
        # note that each of the components inside inception will ALWAYS retain the same width and height, and all the channels are concatenated together thats all
        self.incep1 = InceptionA(64, 16)
        # N, 240, 8, 8
        self.incep2 = InceptionB(240)
        # N, 720, 8, 8
        # inception 3 barely fits CIFAR10, i think this will be the last layer
        self.incep3 = InceptionC(720, 128)
        # N, 768, 8, 8

        # going into the output layer now, last conv layer and then flattening it
        self.out = Sequential(
            # lowering the number of channels
            Conv2d_BN(channels_in = 768, channels_out= 320, kernel_size=1, stride=1, padding=0),
            AdaptiveAvgPool2d(1),
            Dropout(0.5)
        )
        # N, 1024, 1, 1

        # this one will then output it based on the number of classes, based on softmax i guess at this point
        self.fc = Linear(320, class_num)

    def forward(self, x):
        x = self.layer1(x)
        fmap1 = x.clone()
        x = self.layer2(x)
        fmap2 = x.clone()
        x = self.incep1(x)
        fmap3 = x.clone()
        x = self.incep2(x)
        fmap4 = x.clone()
        x = self.incep3(x)
        fmap5 = x.clone()
        x = self.out(x)
        fmap8 = x.clone()
        # this one is to flatten, and retaining the batch size
        x = torch.flatten(x, 1)
        fmap9 = x.clone()
        x = self.fc(x)
        fmap10 = x.clone()

        if self.PCA:
            return x, (fmap1, fmap2, fmap3, fmap3, fmap4, fmap5, fmap8, fmap9, fmap10)
        else:
            return x

class InceptionModel4(Module):
    def __init__(self, channels_in, class_num = 10):
        super(InceptionModel4, self).__init__()
        # remember, i must be able to extract the feature maps of each of the convolution layers, and as such, i must design my network around that as well

        # if this one is false, it will return feature maps. This would be found at the return funtion in the forward function
        self.PCA = False
        # input is N, 3, 32, 32
        self.layer1 = Sequential(
            Conv2d_BN(channels_in = channels_in, channels_out= 32, kernel_size=3, stride=1, padding=1)
            #MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 32, 16, 16
        self.layer2 = Sequential(
            Conv2d_BN(channels_in = 32, channels_out= 64, kernel_size=3, stride=1, padding=1)
            #MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        )
        # N, 64, 8, 8

        # going into the inception layers
        # note that each of the components inside inception will ALWAYS retain the same width and height, and all the channels are concatenated together thats all
        self.incep1 = InceptionA(64, 16)
        # N, 240, 8, 8
        self.incep2 = InceptionB(240)
        # N, 720, 8, 8

        # going into the output layer now, last conv layer and then flattening it
        self.out = Sequential(
            # lowering the number of channels
            Conv2d_BN(channels_in = 720, channels_out= 320, kernel_size=1, stride=1, padding=0),
            AdaptiveAvgPool2d(1),
            Dropout(0.5)
        )
        # N, 1024, 1, 1

        # this one will then output it based on the number of classes, based on softmax i guess at this point
        self.fc = Linear(320, class_num)

    def forward(self, x):
        x = self.layer1(x)
        fmap1 = x.clone()
        x = self.layer2(x)
        fmap2 = x.clone()
        x = self.incep1(x)
        fmap3 = x.clone()
        x = self.incep2(x)
        fmap4 = x.clone()
        x = self.out(x)
        fmap8 = x.clone()
        # this one is to flatten, and retaining the batch size
        x = torch.flatten(x, 1)
        fmap9 = x.clone()
        x = self.fc(x)
        fmap10 = x.clone()

        if self.PCA:
            return x, (fmap1, fmap2, fmap3, fmap3, fmap4, fmap8, fmap9, fmap10)
        else:
            return x

# this is the intermediate modules
class Conv2d_BN(Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride=1, acti=LeakyReLU(0.2, inplace=True)):
        super(Conv2d_BN, self).__init__()
        self.conv2d_bn = Sequential(
            Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            BatchNorm2d(channels_out),
            acti
        )

    def forward(self, x):
        return self.conv2d_bn(x)

class InceptionA(Module):
    def __init__(self, channels_in, pool_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 64, 1, stride=1, padding=0)  # 64 channels
        self.branch5x5 = Sequential(
            Conv2d_BN(channels_in, 48, 1, stride=1, padding=0),
            Conv2d_BN(48, 64, 5, stride=1, padding=2)
        )  # 64 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, stride=1, padding=0),
            Conv2d_BN(64, 96, 3, stride=1, padding=1),
            Conv2d_BN(96, 96, 3, stride=1, padding=1)
        )  # 96 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, pool_channels, 1, stride=1, padding=0)
        )  # pool_channels

    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch5x5(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 64 + 64 + 96 + pool_channels
        return torch.cat(outputs, 1)

class InceptionB(Module):
    def __init__(self, channels_in):
        super(InceptionB, self).__init__()
        self.branch3x3 = Conv2d_BN(channels_in, 384, 3, stride=2, padding=1)  # 384 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, padding=0),
            Conv2d_BN(64, 96, 3, padding=1),
            Conv2d_BN(96, 96, 3, stride=2, padding=1)
        )  # 96 channels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 384 + 96 + channels_in
        return torch.cat(outputs, 1)

class InceptionC(Module):
    def __init__(self, channels_in, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)  # 192 channels
        self.branch7x7 = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, 192, (7, 1), stride=1, padding=(3, 0))
        )  # 192 channels
        self.branch7x7dbl = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, 192, (1, 7), stride=1, padding=(0, 3))
        )  # 192 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels

    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch7x7(x), self.branch7x7dbl(x), self.branch_pool(x)]
        # 192 + 192 + 192 + 192 = 768 channels
        return torch.cat(outputs, 1)

class InceptionD(Module):
    def __init__(self, channels_in):
        super(InceptionD, self).__init__()
        self.branch3x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 320, 3, stride=2, padding=1)
        )  # 320 channels
        self.branch7x7x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 192, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(192, 192, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(192, 192, 3, stride=2, padding=1)
        )  # 192 chnnels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)]
        # 320 + 192 + channels_in
        return torch.cat(outputs, 1)

class InceptionE(Module):
    def __init__(self, channels_in):
        super(InceptionE, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 320, 1, stride=1, padding=0)  # 320 channels

        self.branch3x3_1 = Conv2d_BN(channels_in, 384, 1, stride=1, padding=0)
        self.branch3x3_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels

        self.branch3x3dbl_1 = Sequential(
            Conv2d_BN(channels_in, 448, 1, stride=1, padding=0),
            Conv2d_BN(448, 384, 3, stride=1, padding=1)
        )
        self.branch3x3dbl_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3dbl_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels

        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = torch.cat([self.branch3x3dbl_2a(branch3x3dbl), self.branch3x3dbl_2b(branch3x3dbl)], 1)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # 320 + 768 + 768 + 192 = 2048 channels
        return torch.cat(outputs, 1)
# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.leakyrelu(x)
        return x


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)

        self.conv2_1 = ConvBlock(64, 192, kernel_size=3, padding=1)

        self.conv3_1 = ConvBlock(192, 128, kernel_size=1)
        self.conv3_2 = ConvBlock(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = ConvBlock(256, 256, kernel_size=1)
        self.conv3_4 = ConvBlock(256, 512, kernel_size=3, padding=1)

        self.conv4_1 = ConvBlock(512, 256, kernel_size=1)
        self.conv4_2 = ConvBlock(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = ConvBlock(512, 512, kernel_size=1)
        self.conv4_4 = ConvBlock(512, 1024, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBlock(1024, 512, kernel_size=1)
        self.conv5_2 = ConvBlock(512, 1024, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool(x)

        for _ in range(4):
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool(x)

        for _ in range(2):
            x = self.conv5_1(x)
            x = self.conv5_2(x)
        return x


class YOLO(nn.Module):
    def __init__(self, darknet, n_classes, n_grids=7, n_bboxes=2):
        super().__init__()

        self.darknet = darknet
        self.n_classes = n_classes
        self.n_grids = n_grids
        self.n_bboxes = n_bboxes

        self.conv5_3 = ConvBlock(1024, 1024, kernel_size=3, padding=1)
        self.conv5_4 = ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1)

        self.conv6_1 = ConvBlock(1024, 1024, kernel_size=3, padding=1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.linear1 = nn.Linear(1024 * n_grids * n_grids, 4096)
        self.linear2 = nn.Linear(4096, n_grids * n_grids * (5 * n_bboxes + n_classes))

        self.dropout = nn.Dropout(0.5)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.darknet(x)

        x = self.conv5_3(x)
        x = self.conv5_4(x)

        x = self.conv6_1(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.leakyrelu(x)

        x = self.linear2(x)
        x = x.view((b, (5 * self.n_bboxes + self.n_classes), self.n_grids, self.n_grids))
        x = self.relu(x)
        # x = self.sigmoid(x)

        # x[:, 10:, ...] = F.softmax(x[:, 10:, ...], dim=1)
        return x


if __name__ == "__main__":
    darknet = Darknet()
    yolo = YOLO(darknet=darknet, n_classes=20)
    x = torch.randn((4, 3, 448, 448))
    yolo(x).shape
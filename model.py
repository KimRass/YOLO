# References
    # https://github.com/motokimura/model_v1_pytorch/blob/master/model_v1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=config.LEAKY_RELU_SLOPE)
        return x


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3) # "7×7×64-s-2"

        self.conv2_1 = ConvBlock(64, 192, kernel_size=3, padding=1) # "3×3×192"

        self.conv3_1 = ConvBlock(192, 128, kernel_size=1) # "1×1×128"
        self.conv3_2 = ConvBlock(128, 256, kernel_size=3, padding=1)  # "3×3×256"
        self.conv3_3 = ConvBlock(256, 256, kernel_size=1) # "1×1×256"
        self.conv3_4 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"

        self.conv4_1 = ConvBlock(512, 256, kernel_size=1) # "1×1×256"
        self.conv4_2 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"
        self.conv4_3 = ConvBlock(512, 512, kernel_size=1) # "1×1×512"
        self.conv4_4 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv5_1 = ConvBlock(1024, 512, kernel_size=1) # "1×1×512"
        self.conv5_2 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"


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

        for _ in range(4): # "×4"
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool(x)

        for _ in range(2): # "×2"
            x = self.conv5_1(x)
            x = self.conv5_2(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.darknet = Darknet()

        # "We use a linear activation function for the final layer and all other layers use
        # the leaky rectified linear activation."
        self.conv5_3 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv5_4 = ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1) # "3×3×1024-s-2"

        self.conv6_1 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv6_2 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"

        self.proj1 = nn.Linear(1024 * config.N_CELLS * config.N_CELLS, 4096)
        self.proj2 = nn.Linear(
            4096, config.N_CELLS * config.N_CELLS * (5 * config.N_BBOXES + len(config.VOC_CLASSES))
        )

        # "A dropout layer with rate = .5 after the first connected layer prevents co-adaptation
        # between layers"
        self.drop = nn.Dropout(0.5)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.darknet(x)

        x = self.conv5_3(x)
        x = self.conv5_4(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
    
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.proj1(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.proj2(x)
        x = self.relu(x)

        x = x.view((-1, (5 * config.N_BBOXES + len(config.VOC_CLASSES)), config.N_CELLS, config.N_CELLS))
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn((4, 3, 448, 448))
    out = model(x)
    out.shape
    
# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_RELU_SLOPE = 0.1


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE)
        return x


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3) # "7×7×64-s-2"
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv2_1 = ConvBlock(64, 192, kernel_size=3, padding=1) # "3×3×192"
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv3_1 = ConvBlock(192, 128, kernel_size=1) # "1×1×128"
        self.conv3_2 = ConvBlock(128, 256, kernel_size=3, padding=1)  # "3×3×256"
        self.conv3_3 = ConvBlock(256, 256, kernel_size=1) # "1×1×256"
        self.conv3_4 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv4_1 = ConvBlock(512, 256, kernel_size=1) # "1×1×256"
        self.conv4_2 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"
        self.conv4_3 = ConvBlock(512, 512, kernel_size=1) # "1×1×512"
        self.conv4_4 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv5_1 = ConvBlock(1024, 512, kernel_size=1) # "1×1×512"
        self.conv5_2 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool3(x)

        for _ in range(4): # "×4"
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool4(x)

        for _ in range(2): # "×2"
            x = self.conv5_1(x)
            x = self.conv5_2(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, n_cells=7, n_bboxes=2, n_classes=20):
        super().__init__()

        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_classes = n_classes

        self.darknet = Darknet()

        self.conv5_3 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv5_4 = ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1) # "3×3×1024-s-2"

        self.conv6_1 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv6_2 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"

        self.proj1 = nn.Linear(1024 * n_cells * n_cells, 4096)
        self.proj2 = nn.Linear(
            4096, n_cells * n_cells * (5 * n_bboxes + n_classes)
        )
        # "A dropout layer with rate = .5 after the first connected layer prevents co-adaptation
        # between layers"
        self.drop = nn.Dropout(0.5)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.darknet(x)

        x = self.conv5_3(x)
        x = self.conv5_4(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
    
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.proj1(x)
        x = self.drop(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE)
        x = self.proj2(x)
        x = torch.sigmoid(x)
        # "We use a linear activation function for the final layer and all other layers use
        # the leaky rectified linear activation."

        print(x.shape)
        x = x.view(
            (-1, (5 * self.n_bboxes + self.n_classes), self.n_cells, self.n_cells),
        )
        return x


if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn((4, 3, 448, 448))
    out = model(x)
    out.shape

# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(
            self,
            img_size=448,
            n_cells=7,
            n_bboxes=2,
            n_classes=20,
            lamb_coord=5,
            lamb_noobj=0.5,
        ):
        super().__init__()

        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_classes = n_classes
        self.lamb_coord = lamb_coord
        self.lamb_noobj = lamb_noobj

        self.cell_size = img_size // n_cells

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

        x = x.view(
            (-1, (5 * self.n_bboxes + self.n_classes), self.n_cells, self.n_cells),
        )
        return x

    def decode(self, x):
        bbox = x.clone()

        bbox[:, (2, 7), ...] *= self.img_size # w
        bbox[:, (3, 8), ...] *= self.img_size # h

        bbox[:, (0, 5), ...] *= self.cell_size # x
        bbox[:, (0, 5), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(0)
        bbox[:, (1, 6), ...] *= self.cell_size # y
        bbox[:, (1, 6), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(1)

        x1 = (bbox[:, (0, 5), ...] - bbox[:, (2, 7), ...] / 2).round()
        y1 = (bbox[:, (1, 6), ...] - bbox[:, (3, 8), ...] / 2).round()
        x2 = (bbox[:, (0, 5), ...] + bbox[:, (2, 7), ...] / 2).round()
        y2 = (bbox[:, (1, 6), ...] + bbox[:, (3, 8), ...] / 2).round()

        bbox[:, (0, 5), ...] = x1
        bbox[:, (1, 6), ...] = y1
        bbox[:, (2, 7), ...] = x2
        bbox[:, (3, 8), ...] = y2
        bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...] = torch.clip(
            bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...], min=0, max=self.img_size
        )

        bbox1 = torch.cat([bbox[:, : 5, ...], bbox[:, 10:, ...]], dim=1)
        bbox1 = rearrange(bbox1, pattern="b c h w -> b (h w) c")

        bbox2 = torch.cat([bbox[:, 5: 10, ...], bbox[:, 10:, ...]], dim=1)
        bbox2 = rearrange(bbox2, pattern="b c h w -> b (h w) c")

        bbox = torch.cat([bbox1, bbox2], dim=1)
        return torch.cat([bbox[..., : 5], bbox[..., 5:]], dim=2)

    def get_loss(self, x, gt):
        out = self(x)
        pred = self.decode(out)

        xy_indices = (0, 1, 5, 6)
        wh_indices = (2, 3, 7, 8)
        conf_indices = (4, 9)
        cls_indices = range(10, 30)

        b, _, _, _ = pred.shape

        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

        obj_mask = (gt[..., 4] == 1)
        noobj_mask = (gt[..., 4] != 1)

        obj_indices = obj_mask.nonzero(as_tuple=True)
        noobj_indices = noobj_mask.nonzero(as_tuple=True)

        ### Coordinate loss
        pred_xy_obj = pred[..., xy_indices][obj_indices]
        gt_xy_obj = gt[..., xy_indices][obj_indices]
        # The 1st term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} \bigg]$$"
        xy_loss = self.lamb_coord * F.mse_loss(pred_xy_obj, gt_xy_obj, reduction="mean")

        pred_wh_obj = pred[..., wh_indices][obj_indices]
        gt_wh_obj = gt[..., wh_indices][obj_indices]
        # The 2nd term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} \bigg]$$"
        wh_loss = self.lamb_coord * F.mse_loss(pred_wh_obj ** 0.5, gt_wh_obj ** 0.5)
        coord_loss = xy_loss + wh_loss

        ### Confidence loss
        pred_conf_obj = pred[..., conf_indices][obj_indices]
        gt_conf_obj = gt[..., conf_indices][obj_indices]
        # The 3rd term; "$$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$"
        conf_loss_obj = F.mse_loss(pred_conf_obj, gt_conf_obj)

        pred_conf_noobj = pred[..., conf_indices][noobj_indices]
        gt_conf_noobj = gt[..., conf_indices][noobj_indices]
        # The 4th term; "$$\lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # 1^{noobj}_{ij} \big( C_{i} - \hat{C}_{i} \big)^{2}$$"
        conf_loss_noobj = self.lamb_noobj * F.mse_loss(pred_conf_noobj, gt_conf_noobj)

        conf_loss = conf_loss_obj + conf_loss_noobj

        ### Classification loss
        pred_cls_obj = pred[..., cls_indices][obj_indices]
        gt_cls_obj = gt[..., cls_indices][obj_indices]
        # The 5th term; "$$\sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes}\
            # \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}$$"
        cls_loss = F.mse_loss(pred_cls_obj, gt_cls_obj)

        loss = coord_loss + conf_loss + cls_loss
        loss /= b
        return loss


if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn((4, 3, 448, 448))
    out = model(x)
    pred = model.decode(out)
    pred.shape
    pred[0, : 4, : 5]

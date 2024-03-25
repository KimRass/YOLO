# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from eval import get_iou

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
            coord_coeff=5,
            noobj_coeff=0.5,
        ):
        super().__init__()

        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_classes = n_classes
        self.coord_coeff = coord_coeff
        self.noobj_coeff = noobj_coeff

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

    def denormalize_xywh(self, norm_xywh):
        # norm_xywh = pred_norm_xywh
        xywh = norm_xywh.clone()
        xywh[:, :, :, 0] *= self.cell_size
        xywh[:, :, :, 1] *= self.cell_size
        xywh[:, :, :, 2] *= self.img_size
        xywh[:, :, :, 3] *= self.img_size
        return xywh

    def xywh_to_ltrb(self, xywh):
        l = torch.clip(xywh[:, :, :, 0] - xywh[:, :, :, 2] / 2, min=0)
        t = torch.clip(xywh[:, :, :, 1] - xywh[:, :, :, 3] / 2, min=0)
        r = torch.clip(xywh[:, :, :, 0] + xywh[:, :, :, 2] / 2, max=self.img_size)
        b = torch.clip(xywh[:, :, :, 1] + xywh[:, :, :, 3] / 2, max=self.img_size)
        return torch.stack([l, t, r, b], dim=3)

    def _get_resp_mask(self, pred_norm_xywh, gt_norm_xywh, obj_mask):
        """
        "$\mathbb{1}^{obj}_{ij}$"; "The $j$th bounding box predictor in cell $i$
        is 'responsible' for that prediction." 

        Args:
            pred_norm_xywh (_type_): _description_
            gt_norm_xywh (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred_xywh = self.denormalize_xywh(pred_norm_xywh)
        pred_ltrb = self.xywh_to_ltrb(pred_xywh)

        gt_xywh = self.denormalize_xywh(gt_norm_xywh)
        gt_ltrb = self.xywh_to_ltrb(gt_xywh)

        iou = get_iou(pred_ltrb, gt_ltrb)
        _, idx_max = torch.max(iou, dim=2, keepdim=False)
        iou_mask = F.one_hot(idx_max[:, :, 0], num_classes=self.n_bboxes)[:, :, :, None].bool()
        return iou_mask * obj_mask.repeat(1, 1, self.n_bboxes, 1)

    def get_coord_loss(self, out, gt_norm_xywh, obj_mask):
        pred_norm_xywh = rearrange(out[:, 0: 8], pattern="b (n c) h w -> b (h w) c n", n=4)
        resp_mask = self._get_resp_mask(
            pred_norm_xywh=pred_norm_xywh,
            gt_norm_xywh=gt_norm_xywh,
            obj_mask=obj_mask,
        )
        x_mse = (
            resp_mask * F.mse_loss(
                pred_norm_xywh[:, :, :, 0: 1],
                gt_norm_xywh[:, :, :, 0: 1].repeat(1, 1, self.n_bboxes, 1),
                reduction="none",
            )
        ).mean()
        y_mse = (
            resp_mask * F.mse_loss(
                pred_norm_xywh[:, :, :, 1: 2],
                gt_norm_xywh[:, :, :, 1: 2].repeat(1, 1, self.n_bboxes, 1),
                reduction="none",
            )
        ).mean()
        w_mse = (
            resp_mask * F.mse_loss(
                pred_norm_xywh[:, :, :, 2: 3] ** 0.5,
                gt_norm_xywh[:, :, :, 2: 3].repeat(1, 1, self.n_bboxes, 1) ** 0.5,
                reduction="none",
            )
        ).mean()
        h_mse = (
            resp_mask * F.mse_loss(
                pred_norm_xywh[:, :, :, 3: 4] ** 0.5,
                gt_norm_xywh[:, :, :, 3: 4].repeat(1, 1, self.n_bboxes, 1) ** 0.5,
                reduction="none",
            )
        ).mean()
        # print(x_mse, y_mse, w_mse, h_mse)
        return self.coord_coeff * (x_mse + y_mse + w_mse + h_mse)

    def get_conf_loss(self, out, obj_mask):
        pred_conf = rearrange(out[:, 8: 10], pattern="b (n c) h w -> b (h w) c n", n=1)
        obj_conf_loss = (
            obj_mask * F.mse_loss(
                pred_conf, torch.ones_like(pred_conf), reduction="none",
            )
        ).mean()
        noobj_conf_loss = self.noobj_coeff * (
            (~obj_mask) * F.mse_loss(
                pred_conf, torch.zeros_like(pred_conf), reduction="none",
            )
        ).mean() # "$\mathbb{1}^{noobj}_{ij}$"
        return obj_conf_loss + noobj_conf_loss

    def get_cls_loss(self, out, gt_cls_prob):
        pred_cls_prob = rearrange(
            out[:, 10: 30], pattern="b (n c) h w -> b (h w) c n", n=self.n_classes,
        )
        return F.mse_loss(pred_cls_prob, gt_cls_prob, reduction="mean")

    def get_loss(self, x, gt_norm_xywh, gt_cls_prob, obj_mask):
        """
        gt_norm_xywh: [B, N, K, 4]
        gt_cls_prob: [B, N, K, 1]
            L: n_bboxes (2)
            K: n_bboxes_per_cell (1)
        """
        out = self(x)
        # pred_sel_norm_xywh = torch.take_along_dim(pred_norm_xywh, indices=idx_max[:, :, :, None], dim=2)
        coord_loss = self.get_coord_loss(
            out, gt_norm_xywh=gt_norm_xywh, obj_mask=obj_mask,
        )
        conf_loss = self.get_conf_loss(out, obj_mask=obj_mask)
        cls_loss = self.get_cls_loss(out, gt_cls_prob=gt_cls_prob)
        print(coord_loss, conf_loss, cls_loss)
        return coord_loss + conf_loss + cls_loss

if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn((4, 3, 448, 448))
    loss = model.get_loss(
        x, gt_norm_xywh=gt_norm_xywh, gt_cls_prob=gt_cls_prob, obj_mask=obj_mask,
    )
    loss

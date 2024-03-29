# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from eval import get_iou
from utils import VOC_CLASSES, COLORS, image_to_grid, to_pil, denorm

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
    
        layers = list()
        layers.extend(
            [
                ConvBlock(3, 64, kernel_size=7, stride=2, padding=3), # "7×7×64-s-2", conv1_1
                nn.MaxPool2d(kernel_size=2, stride=2), # "2×2×s-2"
                ConvBlock(64, 192, kernel_size=3, padding=1), # "3×3×192", conv2_1
                nn.MaxPool2d(kernel_size=2, stride=2), # "2×2×s-2"
                ConvBlock(192, 128, kernel_size=1), # "1×1×128", conv3_1
                ConvBlock(128, 256, kernel_size=3, padding=1),  # "3×3×256", conv3_2
                ConvBlock(256, 256, kernel_size=1), # "1×1×256", conv3_3
                ConvBlock(256, 512, kernel_size=3, padding=1), # "3×3×512", conv3_4
                nn.MaxPool2d(kernel_size=2, stride=2), # "2×2×s-2"        
            ]
        )
        for _ in range(4): # "×4"
            layers.extend(
                [
                    ConvBlock(512, 256, kernel_size=1), # "1×1×256", conv4_1
                    ConvBlock(256, 512, kernel_size=3, padding=1), # "3×3×512", conv4_2
                ]
            )
        layers.extend(
            [
                ConvBlock(512, 512, kernel_size=1), # "1×1×512", conv4_3
                ConvBlock(512, 1024, kernel_size=3, padding=1), # "3×3×1024", conv4_4
                nn.MaxPool2d(kernel_size=2, stride=2), # "2×2×s-2"
            ]
        )
        for _ in range(2): # "×2"
            layers.extend(
                [
                    ConvBlock(1024, 512, kernel_size=1), # "1×1×512", conv5_1
                    ConvBlock(512, 1024, kernel_size=3, padding=1), # "3×3×1024", conv5_2
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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

        # "We use a linear activation function for the final layer and all other layers use
        # the leaky rectified linear activation."
        self.conv_block = nn.Sequential(
            ConvBlock(1024, 1024, 3, 1, 1), # "3×3×1024", conv5_3
            ConvBlock(1024, 1024, 3, 2, 1), # "3×3×1024-s-2", conv5_4
            ConvBlock(1024, 1024, 3, 1, 1), # "3×3×1024", covn_6_1
            ConvBlock(1024, 1024, 3, 1, 1), # "3×3×1024", conv6_2
        )
        self.linear_block = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            nn.Linear(1024 * n_cells * n_cells, 4096),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            # "A dropout layer with rate = .5 after the first connected layer
            # prevents co-adaptation between layers."
            nn.Dropout(0.5),
            nn.Linear(
                4096, n_cells * n_cells * (5 * n_bboxes + n_classes)
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.darknet(x)
        x = self.conv_block(x)
        x = self.linear_block(x)
        return x.view(
            (-1, (5 * self.n_bboxes + self.n_classes), self.n_cells, self.n_cells),
        )

    def denormalize_xywh(self, norm_xywh):
        xywh = norm_xywh.clone()
        ori_shape = xywh.shape

        xywh[..., 0: 2] *= self.cell_size
        xywh = xywh.view(xywh.size(0), self.n_cells, self.n_cells, xywh.size(2), 4)
        xywh[..., 0] += torch.arange(
            0, self.img_size, self.cell_size, device=norm_xywh.device,
        )[None, :, None, None].repeat(1, 1, self.n_cells, 1)
        xywh[..., 1] += torch.arange(
            0, self.img_size, self.cell_size, device=norm_xywh.device,
        )[None, None, :, None].repeat(1, self.n_cells, 1, 1)
        xywh = xywh.view(ori_shape)

        xywh[..., 2: 4] *= self.img_size
        return xywh

    def xywh_to_ltrb(self, xywh):
        l = torch.clip(xywh[..., 0] - (xywh[..., 2] / 2), min=0)
        t = torch.clip(xywh[..., 1] - (xywh[..., 3] / 2), min=0)
        r = torch.clip(xywh[..., 0] + (xywh[..., 2] / 2), max=self.img_size)
        b = torch.clip(xywh[..., 1] + (xywh[..., 3] / 2), max=self.img_size)
        return torch.stack([l, t, r, b], dim=-1)

    def get_ltrb(self, out):
        pred_norm_xywh = rearrange(out[:, 0: 8], pattern="b (n c) h w -> b (h w) c n", n=4)
        pred_xywh = self.denormalize_xywh(pred_norm_xywh)
        return self.xywh_to_ltrb(pred_xywh)

    def _get_responsibility_mask(self, out, gt_norm_xywh, obj_mask):
        """
        "$\mathbb{1}^{obj}_{ij}$"; "The $j$th bounding box predictor in cell $i$
        is 'responsible' for that prediction." 
        ""$\mathbb{1}^{noobj}_{ij}$"; If invert the return of this function.

        Args:
            pred_norm_xywh (_type_): _description_
            gt_norm_xywh (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred_ltrb = self.get_ltrb(out)

        gt_xywh = self.denormalize_xywh(gt_norm_xywh)
        gt_ltrb = self.xywh_to_ltrb(gt_xywh)

        iou = get_iou(pred_ltrb, gt_ltrb)
        _, idx_max = torch.max(iou, dim=2, keepdim=False)
        iou_mask = F.one_hot(
            idx_max[:, :, 0], num_classes=self.n_bboxes,
        )[:, :, :, None].bool()
        return iou_mask * obj_mask.repeat(1, 1, self.n_bboxes, 1)

    def get_coordinate_loss(self, out, gt_norm_xywh, resp_mask):
        pred_norm_xy = rearrange(out[:, 0: 4], pattern="b (n c) h w -> b (h w) c n", n=2)
        gt_norm_xy = gt_norm_xywh[:, :, :, 0: 2].repeat(1, 1, 2, 1).detach()
        xy_loss = F.mse_loss(
            pred_norm_xy,
            torch.where(resp_mask, gt_norm_xy, pred_norm_xy),
            reduction="mean",
        )
        pred_norm_wh = rearrange(out[:, 4: 8], pattern="b (n c) h w -> b (h w) c n", n=2)
        gt_norm_wh = gt_norm_xywh[:, :, :, 2: 4].repeat(1, 1, 2, 1).detach()
        wh_loss = F.mse_loss(
            pred_norm_wh,
            torch.where(resp_mask, gt_norm_wh, pred_norm_wh),
            reduction="mean",
        )
        return self.coord_coeff * (xy_loss + wh_loss)

    @staticmethod
    def get_confidence(out):
        return rearrange(out[:, 8: 10], pattern="b (n c) h w -> b (h w) c n", n=1)

    def get_confidence_loss(self, out, resp_mask):
        pred_conf = self.get_confidence(out)
        obj_conf_loss = F.mse_loss(
            pred_conf,
            torch.where(resp_mask, torch.ones_like(pred_conf), pred_conf),
            reduction="mean",
        )
        noobj_conf_loss = self.noobj_coeff * F.mse_loss(
            pred_conf,
            torch.where(~resp_mask, torch.zeros_like(pred_conf), pred_conf),
            reduction="mean",
        )
        return obj_conf_loss + noobj_conf_loss

    def get_classification_prob(self, out):
        return F.softmax(
            rearrange(
                out[:, 10: 30], pattern="b (n c) h w -> b (h w) c n", n=self.n_classes,
            ),
            dim=-1,
        )

    def get_classification_loss(self, out, gt_cls_prob):
        pred_cls_prob = self.get_classification_prob(out)
        return F.mse_loss(pred_cls_prob, gt_cls_prob, reduction="mean")

    def get_loss(self, image, gt_norm_xywh, gt_cls_prob, obj_mask):
        """
        gt_norm_xywh: Tensor of shape (B, N, K, 4).
        gt_cls_prob: Tensor of shape (B, N, K, 1).
            L: `self.n_bboxes` (equals to 2)
            K: `self.n_bboxes_per_cell` (equals to 1)
        """
        out = self(image)

        resp_mask = self._get_responsibility_mask(
            out=out,
            gt_norm_xywh=gt_norm_xywh,
            obj_mask=obj_mask,
        )
        coord_loss = self.get_coordinate_loss(
            out, gt_norm_xywh=gt_norm_xywh, resp_mask=resp_mask,
        )
        conf_loss = self.get_confidence_loss(out, resp_mask=resp_mask)
        cls_loss = self.get_classification_loss(out, gt_cls_prob=gt_cls_prob)
        return coord_loss + conf_loss + cls_loss

    @staticmethod
    def to_uint8(image, mean, std):
        return (denorm(image, mean=mean, std=std) * 255).byte()

    @staticmethod
    def prob_to_index(prob):
        return torch.argmax(prob, dim=-1, keepdim=True)

    @torch.inference_mode()
    def draw_gt(
        self,
        image,
        gt_norm_xywh,
        gt_cls_prob,
        obj_mask, mean,
        std,
        padding=1,
    ):
        gt_xywh = self.denormalize_xywh(gt_norm_xywh)
        gt_ltrb = self.xywh_to_ltrb(gt_xywh)
        gt_cls_idx = self.prob_to_index(gt_cls_prob)

        uint8_image = self.to_uint8(image, mean=mean, std=std)
        images = list()
        for batch_idx in range(image.size(0)):
            obj_mask_batch = obj_mask[batch_idx]

            gt_ltrb_batch = gt_ltrb[batch_idx]
            gt_cls_idx_batch = gt_cls_idx[batch_idx]

            boxes = torch.masked_select(
                gt_ltrb_batch, mask=obj_mask_batch,
            ).view(-1, 4)
            cls_indices = torch.masked_select(
                gt_cls_idx_batch, mask=obj_mask_batch,
            ).tolist()
            drawn_image = draw_bounding_boxes(
                image=uint8_image[batch_idx],
                boxes=boxes,
                labels=[VOC_CLASSES[idx] for idx in cls_indices],
                colors=[COLORS[idx] for idx in cls_indices],
                width=2,
            )
            images.append(drawn_image)
        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=padding,
        )
        TF.to_pil_image(grid).show()

    def get_confidence_mask(self, out):
        pred_conf = self.get_confidence(out)
        max_conf, _ = torch.max(pred_conf, dim=2, keepdim=True)
        return (max_conf >= 0.5)

    def get_max_confidence_ltrb(self, out):
        pred_ltrb = self.get_ltrb(out)
        pred_conf = self.get_confidence(out)
        _, max_conf_idx = torch.max(pred_conf, dim=2, keepdim=True)
        return torch.gather(
            pred_ltrb, dim=2, index=max_conf_idx.repeat(1, 1, 1, 4),
        )

    def get_classification_index(self, out):
        pred_cls_prob = self.get_classification_prob(out)
        return torch.argmax(pred_cls_prob, dim=-1, keepdim=True)

    @torch.inference_mode()
    def draw_pred(self, image, out, mean, std, padding=1):
        pred_cls_idx = model.get_classification_index(out)
        conf_mask = model.get_confidence_mask(out)
        pred_ltrb = model.get_max_confidence_ltrb(out)

        uint8_image = self.to_uint8(image, mean=mean, std=std)
        images = list()
        for batch_idx in range(image.size(0)):
            pred_ltrb_batch = pred_ltrb[batch_idx]
            conf_mask_batch = conf_mask[batch_idx]
            pred_cls_idx_batch = pred_cls_idx[batch_idx]

            # ltrb = pred_ltrb_batch[conf_mask_batch.repeat(1, 1, 4)].view(-1, 4)
            boxes = torch.masked_select(
                pred_ltrb_batch, mask=conf_mask_batch,
            ).view(-1, 4)
            # cls_idx = pred_cls_idx_batch[conf_mask_batch].view(-1, 1)
            cls_indices = torch.masked_select(
                pred_cls_idx_batch, mask=conf_mask_batch,
            ).tolist()
            # print(ltrb)
            drawn_image = draw_bounding_boxes(
                image=uint8_image[batch_idx],
                boxes=boxes,
                labels=[VOC_CLASSES[idx] for idx in cls_indices],
                colors=[COLORS[idx] for idx in cls_indices],
                width=2,
            )
            images.append(drawn_image)
        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=padding,
        )
        TF.to_pil_image(grid).show()


if __name__ == "__main__":
    from torch.optim import SGD, AdamW

    DEVICE = torch.device("cuda")

    model = YOLOv1().to(DEVICE)

    # model.draw_gt(
    #     image=image,
    #     gt_norm_xywh=gt_norm_xywh,
    #     gt_cls_prob=gt_cls_prob,
    #     obj_mask=obj_mask,
    #     mean=(0.457, 0.437, 0.404),
    #     std=(0.275, 0.271, 0.284),
    # )

    optim = AdamW(model.parameters(), lr=0.0001)

    image = image.to(DEVICE)
    gt_norm_xywh = gt_norm_xywh.to(DEVICE)
    gt_cls_prob = gt_cls_prob.to(DEVICE)
    obj_mask = obj_mask.to(DEVICE)

    for _ in range(30):
        loss = model.get_loss(
            image=image,
            gt_norm_xywh=gt_norm_xywh,
            gt_cls_prob=gt_cls_prob,
            obj_mask=obj_mask,
        )
        print(f"{loss.item():.3f}")
        # print(gt_cls_prob.sum())
        optim.zero_grad()
        loss.backward()
        optim.step()

    out = model(image)
    model.draw_pred(
        image=image,
        out=out,
        mean=(0.457, 0.437, 0.404),
        std=(0.275, 0.271, 0.284),
    )

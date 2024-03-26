# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2

from eval import get_iou
from utils import image_to_grid

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
            # nn.Sigmoid(),
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

    def out_to_ltrb(self, out):
        pred_norm_xywh = rearrange(out[:, 0: 8], pattern="b (n c) h w -> b (h w) c n", n=4)
        pred_xywh = self.denormalize_xywh(pred_norm_xywh)
        return self.xywh_to_ltrb(pred_xywh)

    def _get_responsibility_mask(self, out, gt_norm_xywh, obj_mask):
        """
        "$\mathbb{1}^{obj}_{ij}$"; "The $j$th bounding box predictor in cell $i$
        is 'responsible' for that prediction." 

        Args:
            pred_norm_xywh (_type_): _description_
            gt_norm_xywh (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred_ltrb = self.out_to_ltrb(out)

        gt_xywh = self.denormalize_xywh(gt_norm_xywh)
        gt_ltrb = self.xywh_to_ltrb(gt_xywh)

        iou = get_iou(pred_ltrb, gt_ltrb)
        _, idx_max = torch.max(iou, dim=2, keepdim=False)
        iou_mask = F.one_hot(
            idx_max[:, :, 0], num_classes=self.n_bboxes,
        )[:, :, :, None].bool()
        return iou_mask * obj_mask

    def get_coord_loss(self, out, gt_norm_xywh, resp_mask):
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
    def out_to_conf(out):
        return rearrange(out[:, 8: 10], pattern="b (n c) h w -> b (h w) c n", n=1)

    def get_conf_loss(self, out, resp_mask):
        pred_conf = self.out_to_conf(out)
        obj_conf_loss = F.mse_loss(
            pred_conf,
            torch.where(resp_mask, 1., 0.),
            reduction="mean",
        )
        noobj_conf_loss = self.noobj_coeff * F.mse_loss(
            pred_conf,
            torch.where(~resp_mask, 1., 0.),
            reduction="mean",
        ) # "$\mathbb{1}^{noobj}_{ij}$"
        return obj_conf_loss + noobj_conf_loss

    def get_cls_loss(self, out, gt_cls_prob):
        pred_cls_prob = rearrange(
            out[:, 10: 30], pattern="b (n c) h w -> b (h w) c n", n=self.n_classes,
        )
        return F.mse_loss(pred_cls_prob, gt_cls_prob, reduction="mean")

    def get_loss(self, image, gt_norm_xywh, gt_cls_prob, obj_mask):
        """
        gt_norm_xywh: [B, N, K, 4]
        gt_cls_prob: [B, N, K, 1]
            L: n_bboxes (2)
            K: n_bboxes_per_cell (1)
        """
        out = self(image)

        resp_mask = self._get_responsibility_mask(
            out=out,
            gt_norm_xywh=gt_norm_xywh,
            obj_mask=obj_mask,
        )
        coord_loss = self.get_coord_loss(
            out, gt_norm_xywh=gt_norm_xywh, resp_mask=resp_mask,
        )
        conf_loss = self.get_conf_loss(out, resp_mask=resp_mask)
        cls_loss = self.get_cls_loss(out, gt_cls_prob=gt_cls_prob)
        return coord_loss + conf_loss + cls_loss

    # def denormalize_xywh(norm_xywh):
    #     xywh = norm_xywh.clone()
    #     xywh[:, :, :, 0] *= 64
    #     xywh[:, :, :, 1] *= 64
    #     xywh[:, :, :, 2] *= 448
    #     xywh[:, :, :, 3] *= 448
    #     return norm_xywh

    @torch.inference_mode()
    def draw_pred(self, image, out):
        import numpy as np
        out = model(image)
        # out.max()
        # pred_norm_xywh = rearrange(out[:, 0: 8], pattern="b (n c) h w -> b (h w) c n", n=4)
        # pred_xywh = model.denormalize_xywh(pred_norm_xywh)
        # pred_xywh.max()
        
        pred_ltrb = model.out_to_ltrb(out)
        pred_conf = model.out_to_conf(out)
        pil_image = image_to_grid(image, n_cols=int(image.size(0) ** 0.5))
        img = np.array(pil_image)
        # pred_ltrb.shape, pred_conf.shape
        # pred_conf[batch_idx].view(98)
        for batch_idx in range(pred_ltrb.size(0)):
            ltrb = pred_ltrb[batch_idx].view(-1, 4)
            conf = pred_conf[batch_idx].reshape(-1, 1)
            # ltrb.shape, conf.shape
            for l, t, r, b in ltrb[(conf >= 0.5).repeat(1, 4)].view(-1, 4):
                l = int(l.item())
                t = int(t.item())
                r = int(r.item())
                b = int(b.item())
                cv2.rectangle(img=img, pt1=(l, t), pt2=(r, b), color=(255, 0, 0), thickness=1)
                cv2.circle(
                    img=img,
                    center=((l + r) // 2, (t + b) // 2),
                    radius=1,
                    color=(255, 0, 0),
                    thickness=2,
                )


if __name__ == "__main__":
    from torch.optim import SGD, AdamW

    DEVICE = torch.device("mps")

    model = YOLOv1().to(DEVICE)
    # optim = SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    optim = AdamW(model.parameters(), lr=0.0001)
    # optim = SGD(model.parameters(), lr=0.0001)

    image = image.to(DEVICE)
    gt_norm_xywh = gt_norm_xywh.to(DEVICE)
    gt_cls_prob = gt_cls_prob.to(DEVICE)
    obj_mask = obj_mask.to(DEVICE)
    # gt_cls_prob.sum(dim=3).nonzero()
    # gt_cls_prob[0, 3, 0, :]
    # image_to_grid(image, n_cols=1).show()

    for _ in range(40):
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


    # gt_norm_xywh.sum()
    # gt_norm_xywh.requires_grad

    # out = model(image)
    # resp_mask = model._get_responsibility_mask(
    #     out=out,
    #     gt_norm_xywh=gt_norm_xywh,
    #     obj_mask=obj_mask,
    # )
    # pred_norm_xy = rearrange(out[:, 0: 4], pattern="b (n c) h w -> b (h w) c n", n=2)
    # gt_norm_xy = gt_norm_xywh[:, :, :, 0: 2].repeat(1, 1, 2, 1)
    # target = torch.where(resp_mask, gt_norm_xy, pred_norm_xy)
    # target.requires_grad = False
    # xy_loss = F.mse_loss(
    #     pred_norm_xy,
    #     target,
    #     reduction="mean",
    # )
    # xy_loss

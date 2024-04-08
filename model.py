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

# "At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall."
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias
            ),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
        )
    
    def forward(self, x):
        return self.layers(x)


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
    
        layers = list()
        layers.extend(
            [
                ConvBlock(3, 64, 7, 2, 3), # "7×7×64-s-2", conv1_1
                nn.MaxPool2d(2, 2), # "2×2×s-2"
                ConvBlock(64, 192, 3, 1, 1), # "3×3×192", conv2_1
                nn.MaxPool2d(2, 2), # "2×2×s-2"
                ConvBlock(192, 128, 1, 1, 0), # "1×1×128", conv3_1
                ConvBlock(128, 256, 3, 1, 1),  # "3×3×256", conv3_2
                ConvBlock(256, 256, 1, 1, 0), # "1×1×256", conv3_3
                ConvBlock(256, 512, 3, 1, 1), # "3×3×512", conv3_4
                nn.MaxPool2d(2, 2), # "2×2×s-2"        
            ]
        )
        for _ in range(4): # "×4"
            layers.extend(
                [
                    ConvBlock(512, 256, 1, 1, 0), # "1×1×256", conv4_1
                    ConvBlock(256, 512, 3, 1, 1), # "3×3×512", conv4_2
                ]
            )
        layers.extend(
            [
                ConvBlock(512, 512, 1, 1, 0), # "1×1×512", conv4_3
                ConvBlock(512, 1024, 3, 1, 1), # "3×3×1024", conv4_4
                nn.MaxPool2d(2, 2), # "2×2×s-2"
            ]
        )
        for _ in range(2): # "×2"
            layers.extend(
                [
                    ConvBlock(1024, 512, 1, 1, 0), # "1×1×512", conv5_1
                    ConvBlock(512, 1024, 3, 1, 1), # "3×3×1024", conv5_2
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

        # "We use a linear activation function for the final layer
        # and all other layers use the leaky rectified linear activation."
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
        """
        x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, p1, p2, ..., pN
        """
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

    def get_norm_xywh(self, out):
        pred_norm_xywh1 = rearrange(
            torch.sigmoid(out[:, 0: 4, :, :]), pattern="b n h w -> b (h w) n",
        )
        pred_norm_xywh2 = rearrange(
            torch.sigmoid(out[:, 5: 9, :, :]), pattern="b n h w -> b (h w) n",
        )
        return torch.stack([pred_norm_xywh1, pred_norm_xywh2], dim=2)

    def get_ltrb(self, out):
        pred_norm_xywh = self.get_norm_xywh(out)
        pred_xywh = self.denormalize_xywh(pred_norm_xywh)
        return self.xywh_to_ltrb(pred_xywh)

    def _get_responsibility_mask(self, out, gt_norm_xywh, obj_mask):
        """
        "$\mathbb{1}^{obj}_{ij}$"; "The $j$th bounding box predictor
        in cell $i$ is 'responsible' for that prediction." 
        "$\mathbb{1}^{noobj}_{ij}$"; You can get this if you invert
        the return of this function.

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
        _, max_iou_idx = torch.max(iou, dim=2, keepdim=False)
        iou_mask = F.one_hot(
            max_iou_idx[:, :, 0], num_classes=self.n_bboxes,
        )[:, :, :, None].bool()
        return iou_mask * obj_mask.repeat(1, 1, self.n_bboxes, 1)

    def get_coordinate_loss(self, out, gt_norm_xywh, resp_mask):
        pred_norm_xywh = self.get_norm_xywh(out)
        pred_norm_xy = pred_norm_xywh[:, :, :, 0: 2]
        gt_norm_xy = gt_norm_xywh[:, :, :, 0: 2].repeat(1, 1, 2, 1)
        xy_loss = torch.sum(
            resp_mask * F.mse_loss(
                pred_norm_xy, gt_norm_xy, reduction="none",
            )
        )
        pred_norm_wh = pred_norm_xywh[:, :, :, 2: 4]
        gt_norm_wh = gt_norm_xywh[:, :, :, 2: 4].repeat(1, 1, 2, 1)
        wh_loss = torch.sum(
            resp_mask * F.mse_loss(
                pred_norm_wh ** 0.5, gt_norm_wh ** 0.5, reduction="none",
            )
        )
        return self.coord_coeff * (xy_loss + wh_loss)

    @staticmethod
    def get_confidence(out):
        """
        Returns: Tensor of shape (B, `n_cells ** 2`, 2, 1)
        """
        return rearrange(
            torch.sigmoid(out[:, (4, 9), :, :]), pattern="b n h w -> b (h w) n",
        )[:, :, :, None]

    def get_confidence_loss(self, out, resp_mask):
        """
        "If no bject exists in that cell, the confidence scores should be zero.
        Otherwise we want the confidence score to equal the intersection over union
        (IOU) between the predicted box and the ground truth."
        "The confidence prediction represents the IOU between the predicted box
        and any ground truth box.
        """
        pred_ltrb = self.get_ltrb(out)
        gt_xywh = self.denormalize_xywh(gt_norm_xywh)
        gt_ltrb = self.xywh_to_ltrb(gt_xywh)
        iou = get_iou(pred_ltrb, gt_ltrb)

        pred_conf = self.get_confidence(out)

        obj_conf_loss = torch.sum(
            resp_mask * F.mse_loss(pred_conf, iou, reduction="none")
        )
        noobj_conf_loss = self.noobj_coeff * torch.sum(
            (~resp_mask) * F.mse_loss(
                pred_conf, torch.zeros_like(pred_conf), reduction="none",
            )
        )
        # print(f"{obj_conf_loss.item():.3f}, {noobj_conf_loss.item():.3f}")
        return obj_conf_loss + noobj_conf_loss

    def get_classification_prob(self, out):
        return F.softmax(
            rearrange(
                out[:, 10: 30, :, :],
                pattern="b n h w -> b (h w) n",
                n=self.n_classes,
            )[:, :, None, :],
            dim=-1,
        )

    def get_classification_loss(self, out, gt_cls_prob):
        pred_cls_prob = self.get_classification_prob(out)
        return torch.sum(
            obj_mask.repeat(1, 1, 1, self.n_classes) * F.mse_loss(
                pred_cls_prob, gt_cls_prob, reduction="none",
            )
        )

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
        print(f"{coord_loss.item():.3f}, {conf_loss.item():.3f}, {cls_loss.item():.3f}")
        return (coord_loss + conf_loss + cls_loss) / image.size(0)

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
        obj_mask,
        mean,
        std,
        padding=1,
    ):
        """
        "At test time we multiply the conditional class probabilities and
        the individual box confidence predictions which gives us
        class-specific confidence scores for each box."
        """
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
                # font=Path(__file__).resolve().parent/"resources/NotoSans_Condensed-Medium.ttf",
                # font_size=14,
            )
            images.append(drawn_image.cpu())
        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=padding,
            pad_value=255,
        )
        TF.to_pil_image(grid).show()

    def get_confidence_mask(self, out, conf_thresh):
        pred_conf = self.get_confidence(out)
        max_conf, _ = torch.max(pred_conf, dim=2, keepdim=True)
        return (max_conf >= conf_thresh)

    def get_max_confidence_ltrb(self, out):
        pred_ltrb = self.get_ltrb(out)
        pred_conf = self.get_confidence(out)
        _, max_conf_idx = torch.max(pred_conf, dim=2, keepdim=True)
        return torch.gather(
            pred_ltrb, dim=2, index=max_conf_idx.repeat(1, 1, 1, 4),
        )

    def _get_classification_index(self, out):
        pred_cls_prob = self.get_classification_prob(out)
        return torch.argmax(pred_cls_prob, dim=-1, keepdim=True)

    @torch.inference_mode()
    def draw_pred(self, image, out, mean, std, conf_thresh=0.5, padding=1):
        pred_ltrb = self.get_max_confidence_ltrb(out.cpu())
        pred_cls_idx = self._get_classification_index(out.cpu())
        conf_mask = self.get_confidence_mask(
            out.cpu(), conf_thresh=conf_thresh,
        )

        uint8_image = self.to_uint8(image, mean=mean, std=std)
        images = list()
        for batch_idx in range(image.size(0)):
            # batch_idx=1
            pred_ltrb_batch = pred_ltrb[batch_idx]
            pred_cls_idx_batch = pred_cls_idx[batch_idx]
            conf_mask_batch = conf_mask[batch_idx]

            boxes = torch.masked_select(
                pred_ltrb_batch, mask=conf_mask_batch,
            ).view(-1, 4)
            # boxes

            cls_indices = torch.masked_select(
                pred_cls_idx_batch, mask=conf_mask_batch,
            ).tolist()
            drawn_image = draw_bounding_boxes(
                image=uint8_image[batch_idx],
                boxes=boxes,
                labels=[VOC_CLASSES[idx] for idx in cls_indices],
                colors=[COLORS[idx] for idx in cls_indices],
                width=2,
            )
            images.append(drawn_image.cpu())
        grid = make_grid(
            torch.stack(images, dim=0),
            nrow=int(image.size(0) ** 0.5),
            padding=padding,
            pad_value=255,
        )
        TF.to_pil_image(grid).show()


if __name__ == "__main__":
    from torch.optim import SGD, AdamW

    DEVICE = torch.device("cuda")

    model = YOLOv1().to(DEVICE)
    
    image = image.to(DEVICE)
    annot = {k: v.to(DEVICE) for k, v in annot.items()}

    # optim = AdamW(model.parameters(), lr=0.0001)
    # for _ in range(60):
    #     loss = model.get_loss(
    #         image=image,
    #         gt_norm_xywh=gt_norm_xywh,
    #         gt_cls_prob=gt_cls_prob,
    #         obj_mask=obj_mask,
    #     )
    #     # print(f"{loss.item():.3f}")
    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()

    out = model(image)
    out.shape

    
    # mean=(0.457, 0.437, 0.404)
    # std=(0.275, 0.271, 0.284)
    # conf_thresh = 0.5
    # padding=1

    # pred_ltrb = model.get_max_confidence_ltrb(out.cpu())
    # pred_cls_idx = model._get_classification_index(out.cpu())
    # conf_mask = model.get_confidence_mask(
    #     out.cpu(), conf_thresh=conf_thresh,
    # )
    # obj_mask.sum(), conf_mask.sum()

    # uint8_image = model.to_uint8(image, mean=mean, std=std)
    # images = list()
    # for batch_idx in range(image.size(0)):
    #     # batch_idx=1
    #     pred_ltrb_batch = pred_ltrb[batch_idx]
    #     pred_cls_idx_batch = pred_cls_idx[batch_idx]
    #     conf_mask_batch = conf_mask[batch_idx]
    #     conf_mask_batch.sum()

    #     boxes = torch.masked_select(
    #         pred_ltrb_batch, mask=conf_mask_batch,
    #     ).view(-1, 4)
    #     boxes

    #     cls_indices = torch.masked_select(
    #         pred_cls_idx_batch, mask=conf_mask_batch,
    #     ).tolist()
    #     drawn_image = draw_bounding_boxes(
    #         image=uint8_image[batch_idx],
    #         boxes=boxes,
    #         labels=[VOC_CLASSES[idx] for idx in cls_indices],
    #         colors=[COLORS[idx] for idx in cls_indices],
    #         width=2,
    #     )
    #     images.append(drawn_image.cpu())
    # grid = make_grid(
    #     torch.stack(images, dim=0),
    #     nrow=int(image.size(0) ** 0.5),
    #     padding=padding,
    #     pad_value=255,
    # )
    # TF.to_pil_image(grid).show()
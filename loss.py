# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

# For all predicted boxes that are not matched with a ground truth box, it is minimising the objectness confidence, but ignoring the box coordinates and class probabilities.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from einops import rearrange

from model import Darknet, YOLO
from voc2012 import Transform, VOC2012Dataset
from image_utils import resize_image

np.set_printoptions(precision=3, suppress=True)


# def get_masks(gt):
#     obj_mask = gt[:, 4, :, :] > 0
#     obj_mask = obj_mask.unsqueeze(1).expand_as(gt)
#     noobj_mask = torch.logical_not(obj_mask)
#     return obj_mask, noobj_mask

N_BBOXES = 2
IMG_SIZE = 448
N_CELLS = 7
CELL_SIZE = IMG_SIZE // N_CELLS


def get_gt_bbox(gt, batch_idx, y_idx, x_idx):
    x_, y_, w_, h_ = gt[batch_idx, : 4, y_idx, x_idx]
    x = CELL_SIZE * x_idx + x_ * CELL_SIZE
    y = CELL_SIZE * y_idx + y_ * CELL_SIZE
    w = w_ * IMG_SIZE
    h = h_ * IMG_SIZE

    x1 = x - w
    y1 = y - h
    x2 = x + w
    y2 = y + h
    gt_bbox = torch.Tensor([[x1, y1, x2, y2]])
    return gt_bbox


def decode(pred):
    bboxes = pred.clone()

    # bboxes = bboxes[:, (0, 1, 2, 3, 5, 6, 7, 8), ...]
    bboxes = bboxes[:, : 10, ...]
    # bboxes = bboxes.view(-1, N_BBOXES, 4, N_CELLS, N_CELLS)
    bboxes = bboxes.view(-1, N_BBOXES, 5, N_CELLS, N_CELLS)
    bboxes = bboxes.permute(0, 1, 3, 4, 2)

    bboxes[..., 2] *= IMG_SIZE # w
    bboxes[..., 3] *= IMG_SIZE # h

    bboxes[..., 0] *= CELL_SIZE # x
    bboxes[..., 0] += torch.linspace(0, IMG_SIZE - CELL_SIZE, N_CELLS).unsqueeze(0)
    bboxes[..., 1] *= CELL_SIZE # y
    bboxes[..., 1] += torch.linspace(0, IMG_SIZE - CELL_SIZE, N_CELLS).unsqueeze(1)

    x1 = bboxes[..., 0] - bboxes[..., 2] / 2
    y1 = bboxes[..., 1] - bboxes[..., 3] / 2
    x2 = bboxes[..., 0] + bboxes[..., 2] / 2
    y2 = bboxes[..., 1] + bboxes[..., 3] / 2
    conf = bboxes[..., 4]
    # bboxes = torch.stack([x1, y1, x2, y2], dim=4)
    bboxes = torch.stack([x1, y1, x2, y2, conf], dim=4)
    
    bboxes = rearrange(bboxes, pattern="b c h w k -> b (c h w) k")
    return bboxes


class Yolov1Loss(nn.Module):
    def __init__(self, lamb_coord=5, lamb_noobj=0.5, N_BBOXES=2, n_classes=20):
        super().__init__()

        self.lamb_coord = lamb_coord
        self.lamb_noobj = lamb_noobj
        self.N_BBOXES = N_BBOXES
        self.n_classes = n_classes

    def _get_confidence_loss(pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)
        
        obj_mask = (gt[..., 4] == 1)
        obj_indices = obj_mask.nonzero(as_tuple=True)
        noobj_mask = (gt[..., 4] != 1)
        noobj_indices = noobj_mask.nonzero(as_tuple=True)
        
        ### Confidence loss
        obj_pred_conf = pred[..., (4, 9)][obj_indices]
        # $$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$
        obj_conf_loss = mse(obj_pred_conf, torch.ones_like(obj_pred_conf))
        
        noobj_pred_conf = pred[..., (4, 9)][noobj_indices]
        noobj_conf_loss = lamb_noobj * mse(noobj_pred_conf, torch.zeros_like(noobj_pred_conf))

        conf_loss = obj_conf_loss + noobj_conf_loss
        return conf_loss
    
    def forward(self, gt, pred):
        lamb_coord=5
        lamb_noobj=0.5
        n_classes=20
        n_channels = 5 * N_BBOXES + n_classes
        
        mse = nn.MSELoss(reduction="sum")


        # Localization loss
        IMG_SIZE = 448
        N_CELLS = 7
        CELL_SIZE = IMG_SIZE // N_CELLS

        pred = torch.randn(2, 30, 7, 7)
        pred = pred.clip(0, 1)

        b, _, _, _ = pred.shape
        batched_pred_bboxes = decode(pred)
        batched_gt_bboxes = decode(gt)
        for i in range(b):
            pred_bboxes = batched_pred_bboxes[i]
            gt_bboxes = batched_gt_bboxes[i]
            gt_bboxes = torch.unique(gt_bboxes[gt_bboxes[:, 4] == 1], dim=0)
            gt_bboxes
            pred_bboxes.shape
            
            iou = box_iou(pred_bboxes[:, : 4], gt_bboxes[:, : 4])
            iou.shape
            max_iou, argmax = torch.max(iou, dim=0)
            max_iou, argmax
            iou.shape


        # pred.shape, gt.shape
        # obj_mask = (gt[:, 4, ...] == 1)
        # obj_indices = obj_mask.nonzero(as_tuple=True)
        # # noobj_mask = (gt[:, 4, ...] != 1)
        # # noobj_indices = noobj_mask.nonzero(as_tuple=True)
        # b, _, _, _ = pred.shape

        # mask = torch.zeros(size=(b, N_CELLS ** 2 * N_BBOXES))
        # for batch_idx, y_idx, x_idx in zip(*obj_indices):
        #     gt_bbox = get_gt_bbox(gt=gt, batch_idx=batch_idx, y_idx=y_idx, x_idx=x_idx)
        #     pred_bboxes = batched_pred_bboxes[batch_idx, :, :]

        #     ious = box_iou(gt_bbox, pred_bboxes)
        #     max_iou, argmax = torch.max(ious, dim=1)

        #     mask[batch_idx, argmax] = 1
        # pred
        # batched_pred_bboxes.shape
        # gt.shape
            






# def tensor_to_array(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
#     img *= std
#     img += mean
#     img *= 255.0
#     img = np.clip(img, 0, 255).astype("uint8")
#     return img


def get_whether_each_predictor_is_responsible(gt, pred):
    gt = torch.randn((1, 30, 7, 7))
    pred = torch.randn((1, 30, 7, 7))
    b, _, h, w = gt.shape
    gt_bboxes = gt[:, : 4, ...].permute(0, 2, 3, 1).reshape((-1, 4))
    gt_bboxes[:, 2] += gt_bboxes[:, 0]
    gt_bboxes[:, 3] += gt_bboxes[:, 1]

    pred_bboxes1 = pred[:, : 4,...].permute(0, 2, 3, 1).reshape((-1, 4))
    pred_bboxes1[:, 2] += pred_bboxes1[:, 0]
    pred_bboxes1[:, 3] += pred_bboxes1[:, 1]

    pred_bboxes2 = pred[:, 5: 9,...].permute(0, 2, 3, 1).reshape((-1, 4))
    pred_bboxes2[:, 2] += pred_bboxes2[:, 0]
    pred_bboxes2[:, 3] += pred_bboxes2[:, 1]
    
    ious1 = generalized_box_iou_loss(boxes1=gt_bboxes, boxes2=pred_bboxes1)
    ious2 = generalized_box_iou_loss(boxes1=gt_bboxes, boxes2=pred_bboxes2)
    stacked = torch.stack([ious1, ious2], dim=1)
    argmax = torch.argmax(stacked, dim=1).reshape((b, h, w))

    is_resp = F.one_hot(argmax, num_classes=2).permute(0, 3, 1, 2).float()
    is_resp[:, 0, ...] *= gt[:, 4, ...]
    is_resp[:, 1, ...] *= gt[:, 9, ...]
    # is_resp = torch.concat([onehot[:, k: k + 1, ...].repeat(1, 5, 1, 1) for k in range(N_BBOXES)], dim=1)
    # is_resp.shape
    return is_resp


def get_whether_object_appear_in_each_cell(is_resp):
    appears = torch.max(is_resp, dim=1)[0].unsqueeze(1)
    return appears


class Yolov1Loss(nn.Module):
    def __init__(self, lamb_coord=5, lamb_noobj=0.5, N_BBOXES=2, n_classes=20):
        super().__init__()

        self.lamb_coord = lamb_coord
        self.lamb_noobj = lamb_noobj
        self.N_BBOXES = N_BBOXES
        self.n_classes = n_classes
    
    def forward(self, gt, pred):
        # There are usually many more predicted boxes than ground truth boxes.
        # Very simply YOLO matchs the ground truth to whichever predicted box has the current highest IoU.
        # `is_resp`: Evaluates to 1 when the $j$th box from the $i$th cell is matched to any object, and 0 otherwise.
        # `1 - is_resp`: The opposite to `is_resp`.
        # `appears`: Evaluates to 1 when either of the boxes associated with that cell are matched.
        is_resp = get_whether_each_predictor_is_responsible(gt=gt, pred=pred)
        appears = get_whether_object_appear_in_each_cell(is_resp)
        
        # Localization loss
        # x, y
        xy_sse = F.mse_loss(gt[:, (0, 1, 5, 6), ...], pred[:, (0, 1, 5, 6), ...], reduction="none")
        xy_sse *= torch.concat([is_resp[:, k: k + 1, ...].repeat(1, 2, 1, 1) for k in range(self.N_BBOXES)], dim=1)
        xy_sse *= self.lamb_coord
        # w, h
        wh_sse = F.mse_loss(gt[:, (2, 3, 7, 8), ...] ** 0.5, pred[:, (2, 3, 7, 8), ...] ** 0.5, reduction="none")
        wh_sse *= torch.concat([is_resp[:, k: k + 1, ...].repeat(1, 2, 1, 1) for k in range(self.N_BBOXES)], dim=1)
        wh_sse *= self.lamb_coord
        loc_loss = xy_sse.sum() + wh_sse.sum()

        ### Confidence loss
        conf_sse1 = F.mse_loss(gt[:, (4, 9), ...], pred[:, (4, 9), ...], reduction="none")
        conf_sse1 *= is_resp
        conf_sse2 = F.mse_loss(gt[:, (4, 9), ...], pred[:, (4, 9), ...], reduction="none")
        conf_sse2 *= 1 - is_resp
        conf_sse2 *= self.lamb_noobj
        conf_loss = conf_sse1.sum() + conf_sse2.sum()

        ### Classification loss
        cls_sse = F.mse_loss(gt[:, 10:, ...], pred[:, 10:, ...], reduction="none")
        cls_sse *= appears.repeat(1, self.n_classes, 1, 1)
        cls_loss = cls_sse.sum()
        
        loss = loc_loss + conf_loss + cls_loss
        return loss


if __name__ == "__main__":
    darknet = Darknet()
    yolo = YOLO(darknet=darknet, n_classes=20)

    criterion = Yolov1Loss()

    transform = Transform()
    ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True)
    for batch, (image, gt) in enumerate(dl, start=1):
        pred = yolo(image)
        
        criterion = Yolov1Loss()
        loss = criterion(gt, pred)
        loss
# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py
    # https://www.harrysprojects.com/articles/yolov1.html

# For all predicted boxes that are not matched with a ground truth box, it is minimising the objectness confidence, but ignoring the box coordinates and class probabilities.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.ops import generalized_box_iou_loss
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from yolo import Darknet, YOLO
from parepare_voc2012 import Transform, VOC2012Dataset
from image_utils import resize_image

np.set_printoptions(precision=3, suppress=True)


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= std
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def get_whether_each_predictor_is_responsible(gt, pred):
    # b, _, _, _ = pred.shape
    # confs = pred[:, (4, 9), ...]
    # argmax = torch.argmax(confs, dim=1)
    # onehot = F.one_hot(argmax, num_classes=2).permute(0, 3, 1, 2)
    # is_resp = torch.concat(
    #     [onehot[:, k: k + 1, ...].repeat(1, 5, 1, 1) for k in range(n_bboxes)] +\
    #         [torch.ones((b, n_classes, n_cells, n_cells))],
    #     dim=1
    # )
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
    # is_resp = torch.concat([onehot[:, k: k + 1, ...].repeat(1, 5, 1, 1) for k in range(n_bboxes)], dim=1)
    # is_resp.shape
    return is_resp


def get_whether_object_appear_in_each_cell(is_resp):
    appears = torch.max(is_resp, dim=1)[0].unsqueeze(1)
    return appears


class Yolov1Loss(nn.Module):
    def __init__(self, lamb_coord=5, lamb_noobj=0.5, n_bboxes=2, n_classes=20):
        super().__init__()

        self.lamb_coord = lamb_coord
        self.lamb_noobj = lamb_noobj
        self.n_bboxes = n_bboxes
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
        xy_sse *= torch.concat([is_resp[:, k: k + 1, ...].repeat(1, 2, 1, 1) for k in range(self.n_bboxes)], dim=1)
        xy_sse *= self.lamb_coord
        # w, h
        wh_sse = F.mse_loss(gt[:, (2, 3, 7, 8), ...] ** 0.5, pred[:, (2, 3, 7, 8), ...] ** 0.5, reduction="none")
        wh_sse *= torch.concat([is_resp[:, k: k + 1, ...].repeat(1, 2, 1, 1) for k in range(self.n_bboxes)], dim=1)
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
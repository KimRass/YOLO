import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from einops import rearrange

import config
from model import YOLOv1
from voc2012 import VOC2012Dataset

torch.set_printoptions(linewidth=70)



model = YOLOv1()
model.eval()
ckpt_path = "/Users/jongbeomkim/Downloads/yolo_checkpoints/135.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])

ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
dl = DataLoader(
    ds,
    batch_size=1,
    shuffle=False,
    # num_workers=config.N_WORKERS,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)
for image, gt in dl:
    gt_bboxes = decode(gt)
    gt_bboxes = torch.unique(gt_bboxes, dim=1)

    pred = model(image)
    pred_bboxes = decode(pred)
    for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
        pred_bbox = pred_bbox[pred_bbox[:, 4] * pred_bbox[:, 6] >= 0.9]
        pred_bbox = pred_bbox[torch.sort(pred_bbox[:, 4], dim=0, descending=True)[1]]

        iou = box_iou(pred_bbox[:, : 4], pred_bbox[:, : 4])
        pred_bbox = pred_bbox[(torch.triu((iou >= 0.5), diagonal=1).sum(dim=0) == 0)]
        pred_bbox

        gt_bbox = gt_bbox[gt_bbox[:, 4] >= 0.5]
        gt_bbox
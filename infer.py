import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from einops import rearrange

import config
from model import YOLOv1
from voc2012 import VOC2012Dataset

torch.set_printoptions(linewidth=100, sci_mode=False)


def decode(tensor):
    bboxes = tensor.clone()

    bboxes[:, (2, 7), ...] *= config.IMG_SIZE # w
    bboxes[:, (3, 8), ...] *= config.IMG_SIZE # h

    bboxes[:, (0, 5), ...] *= config.CELL_SIZE # x
    bboxes[:, (0, 5), ...] += torch.linspace(0, config.IMG_SIZE - config.CELL_SIZE, config.N_CELLS).unsqueeze(0)
    bboxes[:, (1, 6), ...] *= config.CELL_SIZE # y
    bboxes[:, (1, 6), ...] += torch.linspace(0, config.IMG_SIZE - config.CELL_SIZE, config.N_CELLS).unsqueeze(1)

    bboxes1 = torch.cat([bboxes[:, : 5, ...], bboxes[:, 10:, ...]], dim=1)
    bboxes2 = torch.cat([bboxes[:, 5: 10, ...], bboxes[:, 10:, ...]], dim=1)
    bboxes1 = rearrange(bboxes1, pattern="b c h w -> b (h w) c")
    bboxes2 = rearrange(bboxes2, pattern="b c h w -> b (h w) c")
    bboxes = torch.cat([bboxes1, bboxes2], dim=1)

    max_val, max_idx = torch.max(bboxes[..., 5:], dim=2)
    bboxes = torch.cat([bboxes[..., : 5], max_idx.unsqueeze(-1), max_val.unsqueeze(-1)], dim=2)
    return bboxes


class mAP

model = YOLOv1()
ckpt_path = "/Users/jongbeomkim/Downloads/yolo_checkpoints/135.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])

ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
dl = DataLoader(
    ds,
    batch_size=2,
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

    # pred_bboxes.shape, gt_bboxes.shape
    for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
        pred_bbox = pred_bbox[pred_bbox[:, 4] >= 0.5]
        pred_bbox = pred_bbox[torch.sort(pred_bbox[:, 4], dim=0)[1]]

        gt_bbox = gt_bbox[gt_bbox[:, 4] >= 0.5]
        # pred_bbox.shape, gt_bbox.shape
        pred_bbox
        
        box_iou(pred_bbox[:, : 4], gt_bbox[:, : 4])
        pred_bbox.shape
        pred_bbox.shape

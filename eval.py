# References:
    # https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
    # https://herbwood.tistory.com/2
    # https://herbwood.tistory.com/3

import torch
from collections import defaultdict
import numpy as np


torch.set_printoptions(linewidth=70)


def get_dtype(ltrb):
    if ltrb.device.type == "mps":
        return torch.float32
    else:
        # return torch.float64
        return torch.float32


def get_area(ltrb):
    """
    args:
        ltrb: Tensor of shape (N, 4)
    returns:
        Tensor of shape (N)
    """
    dtype = get_dtype(ltrb)
    return torch.clip(
        ltrb[..., 2] - ltrb[..., 0], min=0
    ) * torch.clip(ltrb[..., 3] - ltrb[..., 1], min=0).to(dtype)


def get_intersection_area(ltrb1, ltrb2):
    """
    args:
        ltrb1: Tensor of shape (N, 4)
        ltrb2: Tensor of shape (M, 4)
    returns:
        Tensor of shape (N, M)
    """
    # ltrb1 = pred_ltrb
    # ltrb2 = gt_ltrb
    dtype = get_dtype(ltrb1)
    l = torch.maximum(ltrb1[..., 0][..., None], ltrb2[..., 0][None, ...])
    t = torch.maximum(ltrb1[..., 1][..., None], ltrb2[..., 1][None, ...])
    r = torch.minimum(ltrb1[..., 2][..., None], ltrb2[..., 2][None, ...])
    b = torch.minimum(ltrb1[..., 3][..., None], ltrb2[..., 3][None, ...])
    return torch.clip(r - l, min=0) * torch.clip(b - t, min=0).to(dtype)


def get_iou(ltrb1, ltrb2):
    """
    args:
        ltrb1: Tensor of shape (N, 4)
        ltrb2: Tensor of shape (M, 4)
    returns:
        Tensor of shape (N, M)
    """
    ltrb1_area = get_area(ltrb1)
    ltrb2_area = get_area(ltrb2)
    intersec_area = get_intersection_area(ltrb1, ltrb2)
    union_area = ltrb1_area[..., None] + ltrb2_area[None, ...] - intersec_area
    return torch.where(union_area == 0, 0., intersec_area / union_area)


def get_smallest_enclosing_area(bbox1, bbox2):
    l = torch.minimum(bbox1[:, 0][:, None], bbox2[:, 0][None, :])
    t = torch.minimum(bbox1[:, 1][:, None], bbox2[:, 1][None, :])
    r = torch.maximum(bbox1[:, 2][:, None], bbox2[:, 2][None, :])
    b = torch.maximum(bbox1[:, 3][:, None], bbox2[:, 3][None, :])
    return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)


def get_giou(bbox1, bbox2):
    bbox1_area = get_area(bbox1)
    bbox2_area = get_area(bbox2)
    intersec_area = get_intersection_area(bbox1, bbox2)
    union_area = bbox1_area[:, None] + bbox2_area[None, :] - intersec_area
    c = get_smallest_enclosing_area(bbox1, bbox2)
    iou = torch.where(union_area == 0, 0, intersec_area / union_area)
    return torch.where(c == 0, -1, iou - ((c - union_area) / c))


def get_accum_prec_and_recall(pred, gt, cls_idx, iou_thresh):
    """
    Args:
        pred: Tensor of shape (B2, 6)
            (Left, Top, Right, Bottom, Confidence, Class)
        gt: Tensor of shape (B1, 5) (Left, Top, Right, Bottom, Class)
    """
    assert gt.size(1) == 5
    assert pred.size(1) == 6

    is_same_cls = (gt[:, 4][:, None] == cls_idx) & (pred[:, 5][None, :] == cls_idx)

    iou = get_iou(gt[:, : 4], pred[:, : 4])
    iou *= 8
    meets_iou_thresh = (iou >= iou_thresh)

    ious = dict()
    for gt_idx, pred_idx in (is_same_cls & meets_iou_thresh).nonzero():
        ious[pred_idx.item()] = (gt_idx.item(), iou[gt_idx, pred_idx].item())
    ious = sorted(ious.items(), key=lambda x: x[1][1], reverse=True)
    ious = {k: v for k, v in ious}

    is_true = defaultdict(bool)
    gt_is_matched = [False] * gt.size(0)
    pred_is_matched = [False] * pred.size(0)
    for pred_idx in range(pred.size(0)):
        if pred_idx in ious:
            gt_idx = ious[pred_idx][0]
            if (not gt_is_matched[gt_idx]) and (not pred_is_matched[pred_idx]):
                is_true[pred_idx] = True
            else:
                is_true[pred_idx] = False
        else:
            is_true[pred_idx] = False

    order = torch.argsort(pred[:, 4], dim=0, descending=True).tolist()
    accum_tp = np.cumsum(np.array(list(is_true.values()))[order])

    accum_prec = accum_tp / range(1, pred.size(0) + 1)
    accum_recall = accum_tp / gt.size(0)
    return accum_prec, accum_recall


def get_ap(pred, gt, cls_idx, iou_thresh):
    accum_prec, accum_recall = get_accum_prec_and_recall(
        pred=pred, gt=gt, cls_idx=cls_idx, iou_thresh=iou_thresh,
    )

    precs = list()
    for recall in np.linspace(0, 1, 11):
        # recall = 0.3
        prec_ge_recall = accum_prec[accum_recall >= recall]
        if not np.any(prec_ge_recall):
            precs.append(0)
        else:
            precs.append(accum_prec[accum_recall >= recall].max())
    return np.sum(precs) / 11


def get_map(pred, gt, n_classes, iou_thresh=0.5):
    sum_ap = 0
    for cls_idx in range(n_classes):
        sum_ap += get_ap(pred=pred, gt=gt, cls_idx=cls_idx, iou_thresh=iou_thresh)
    return sum_ap / n_classes


if __name__ == "__main__":
    img_size = 64
    n_classes = 10
    n = 32
    m = 1024
    iou_thresh = 0.5
    gt = torch.cat(
        [
            torch.randint(0, img_size, size=(n, 4)),
            torch.randint(0, n_classes, size=(n, 1))
        ],
        dim=1,
    )
    pred = torch.cat(
        [
            torch.randint(0, img_size, size=(m, 4)),
            torch.rand(size=(m, 1)),
            torch.randint(0, n_classes, size=(m, 1)),
        ],
        dim=1,
    )
    get_map(pred=pred, gt=gt, n_classes=n_classes, iou_thresh=iou_thresh)

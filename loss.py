# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn

LAMB_COORD = 5
LAMB_NOOBJ = 0.5


class Yolov1Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

        obj_mask = (gt[..., 4] == 1)
        noobj_mask = (gt[..., 4] != 1)

        obj_indices = obj_mask.nonzero(as_tuple=True)
        noobj_indices = noobj_mask.nonzero(as_tuple=True)

        ### Coordinate loss
        pred_xy_obj = pred[..., (0, 1, 5, 6)][obj_indices]
        gt_xy_obj = gt[..., (0, 1, 5, 6)][obj_indices]
        # "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} \bigg]$$"
        xy_loss = LAMB_COORD * self.mse(pred_xy_obj, gt_xy_obj)

        pred_wh_obj = pred[..., (2, 3, 7, 8)][obj_indices]
        gt_wh_obj = gt[..., (2, 3, 7, 8)][obj_indices]
        # "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} \bigg]$$"
        wh_loss = LAMB_COORD * self.mse(pred_wh_obj ** 0.5, gt_wh_obj ** 0.5)
        coord_loss = xy_loss + wh_loss

        ### Confidence loss
        pred_conf_obj = pred[..., (4, 9)][obj_indices]
        gt_conf_obj = gt[..., (4, 9)][obj_indices]
        # "$$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$"
        conf_loss_obj = self.mse(pred_conf_obj, gt_conf_obj)

        pred_conf_noobj = pred[..., (4, 9)][noobj_indices]
        # "$$\lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # 1^{noobj}_{ij} \big( C_{i} - \hat{C}_{i} \big)^{2}$$"
        conf_loss_noobj = LAMB_NOOBJ * self.mse(pred_conf_noobj, torch.zeros_like(pred_conf_noobj))

        conf_loss = conf_loss_obj + conf_loss_noobj

        ### Classification loss
        pred_cls_obj = pred[..., 10:][obj_indices]
        gt_cls_obj = gt[..., 10:][obj_indices]
        cls_loss = self.mse(pred_cls_obj, gt_cls_obj)

        loss = coord_loss + conf_loss + cls_loss
        return loss


if __name__ == "__main__":
    pred = torch.randn(2, 30, 7, 7)
    pred = pred.clip(0, 1)
    di = iter(dl)

    crit = Yolov1Loss()
    image, gt = next(di)
    loss = crit(pred, gt)
    loss

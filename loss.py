# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn

LAMB_COORD = 5
LAMB_NOOBJ = 0.5


def decode(pred):
    bboxes = pred.clone()

    bboxes = bboxes[:, : 10, ...]
    # bboxes = bboxes.view(-1, config.N_BBOXES, 4, config.N_CELLS, config.N_CELLS)
    bboxes = bboxes.view(-1, config.N_BBOXES, 5, config.N_CELLS, config.N_CELLS)
    bboxes = bboxes.permute(0, 1, 3, 4, 2)



    gt[..., 2] *= config.IMG_SIZE # w
    gt[..., 3] *= config.IMG_SIZE # h

    gt[..., 0] *= config.CELL_SIZE # x
    gt[..., 0] += torch.linspace(0, config.IMG_SIZE - config.CELL_SIZE, config.N_CELLS).unsqueeze(0)
    gt[..., 1] *= config.CELL_SIZE # y
    gt[..., 1] += torch.linspace(0, config.IMG_SIZE - config.CELL_SIZE, config.N_CELLS).unsqueeze(1)

    x1 = gt[..., 0] - gt[..., 2] / 2
    y1 = gt[..., 1] - gt[..., 3] / 2
    x2 = gt[..., 0] + gt[..., 2] / 2
    y2 = gt[..., 1] + gt[..., 3] / 2
    # conf = gt[..., 4]
    gt = torch.stack([x1, y1, x2, y2], dim=3)
    gt[0, ..., 3]

    gt = rearrange(gt, pattern="b c h w k -> b (c h w) k")
    return bboxes


class Yolov1Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="sum")

        self.xy_indices = (0, 1, 5, 6)
        self.wh_indices = (2, 3, 7, 8)
        self.conf_indices = (4, 9)
        self.cls_indices = range(10, 30)

    def forward(self, pred, gt):
        # pred.shape, gt.shape
        b, _, _, _ = pred.shape

        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

        obj_mask = (gt[..., 4] == 1)
        noobj_mask = (gt[..., 4] != 1)

        obj_indices = obj_mask.nonzero(as_tuple=True)
        noobj_indices = noobj_mask.nonzero(as_tuple=True)

        ### Coordinate loss
        pred_xy_obj = pred[..., self.xy_indices][obj_indices]
        gt_xy_obj = gt[..., self.xy_indices][obj_indices]
        # pred_xy_obj
        # gt_xy_obj
        # The 1st term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} \bigg]$$"
        xy_loss = LAMB_COORD * self.mse(pred_xy_obj, gt_xy_obj)
        # xy_loss

        pred_wh_obj = pred[..., self.wh_indices][obj_indices]
        gt_wh_obj = gt[..., self.wh_indices][obj_indices]
        # pred_wh_obj ** 0.5
        # gt_wh_obj ** 0.5
        # The 2nd term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} \bigg]$$"
        wh_loss = LAMB_COORD * self.mse(pred_wh_obj ** 0.5, gt_wh_obj ** 0.5)
        coord_loss = xy_loss + wh_loss
        # coord_loss

        ### Confidence loss
        pred_conf_obj = pred[..., self.conf_indices][obj_indices]
        gt_conf_obj = gt[..., self.conf_indices][obj_indices]
        # pred_conf_obj
        # gt_conf_obj
        # The 3rd term; "$$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$"
        conf_loss_obj = self.mse(pred_conf_obj, gt_conf_obj)

        pred_conf_noobj = pred[..., self.conf_indices][noobj_indices]
        gt_conf_noobj = gt[..., self.conf_indices][noobj_indices]
        # gt_conf_noobj
        # The 4th term; "$$\lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # 1^{noobj}_{ij} \big( C_{i} - \hat{C}_{i} \big)^{2}$$"
        conf_loss_noobj = LAMB_NOOBJ * self.mse(pred_conf_noobj, gt_conf_noobj)

        conf_loss = conf_loss_obj + conf_loss_noobj
        # conf_loss_noobj

        ### Classification loss
        pred_cls_obj = pred[..., self.cls_indices][obj_indices]
        gt_cls_obj = gt[..., self.cls_indices][obj_indices]
        # The 5th term; "$$\sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes}\
            # \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}$$"
        cls_loss = self.mse(pred_cls_obj, gt_cls_obj)

        loss = coord_loss + conf_loss + cls_loss
        # coord_loss, conf_loss, cls_loss
        loss /= b
        return loss


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
    dl = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=True, drop_last=True)
    di = iter(dl)
    image, gt = next(di)
    crit = Yolov1Loss()

    for _ in range(100):
        optim.zero_grad()
        
        pred = model(image)
        loss = crit(pred=pred, gt=gt)
        # loss /= 10
        print(loss)
        
        loss.backward()
        optim.step()

    gt[0, 0, ...]
    pred[0, 0, ...]

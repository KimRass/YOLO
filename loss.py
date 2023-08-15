# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn

LAMB_COORD = 5
LAMB_NOOBJ = 0.5


class Yolov1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        mse = nn.MSELoss(reduction="sum")

        xy_indices = (0, 1, 5, 6)
        wh_indices = (2, 3, 7, 8)
        conf_indices = (4, 9)
        cls_indices = range(10, 30)

        b, _, _, _ = pred.shape

        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

        obj_mask = (gt[..., 4] == 1)
        noobj_mask = (gt[..., 4] != 1)

        obj_indices = obj_mask.nonzero(as_tuple=True)
        noobj_indices = noobj_mask.nonzero(as_tuple=True)

        ### Coordinate loss
        pred_xy_obj = pred[..., xy_indices][obj_indices]
        gt_xy_obj = gt[..., xy_indices][obj_indices]
        # The 1st term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} \bigg]$$"
        xy_loss = LAMB_COORD * mse(pred_xy_obj, gt_xy_obj)

        pred_wh_obj = pred[..., wh_indices][obj_indices]
        gt_wh_obj = gt[..., wh_indices][obj_indices]
        # The 2nd term; "$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij}\
            # \bigg[ (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} \bigg]$$"
        wh_loss = LAMB_COORD * mse(pred_wh_obj ** 0.5, gt_wh_obj ** 0.5)
        coord_loss = xy_loss + wh_loss

        ### Confidence loss
        pred_conf_obj = pred[..., conf_indices][obj_indices]
        gt_conf_obj = gt[..., conf_indices][obj_indices]
        # The 3rd term; "$$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$"
        conf_loss_obj = mse(pred_conf_obj, gt_conf_obj)

        pred_conf_noobj = pred[..., conf_indices][noobj_indices]
        gt_conf_noobj = gt[..., conf_indices][noobj_indices]
        # The 4th term; "$$\lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0}\
            # 1^{noobj}_{ij} \big( C_{i} - \hat{C}_{i} \big)^{2}$$"
        conf_loss_noobj = LAMB_NOOBJ * mse(pred_conf_noobj, gt_conf_noobj)

        conf_loss = conf_loss_obj + conf_loss_noobj

        ### Classification loss
        pred_cls_obj = pred[..., cls_indices][obj_indices]
        gt_cls_obj = gt[..., cls_indices][obj_indices]
        # The 5th term; "$$\sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes}\
            # \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}$$"
        cls_loss = mse(pred_cls_obj, gt_cls_obj)

        loss = coord_loss + conf_loss + cls_loss
        loss /= b
        return loss


if __name__ == "__main__":
    model = YOLOv1()
    optim = SGD(
        model.parameters(),
        lr=0.0005,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
    dl = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=True, drop_last=True)
    di = iter(dl)
    
    image, gt = next(di)

    crit = Yolov1Loss()
    for _ in range(1000):
        optim.zero_grad()
        
        pred = model(image)
        
        # loss = crit(pred, gt)
        loss = mse(pred[0, 4, 0, 0], gt[0, 4, 0, 0])
        loss.backward()
        print(loss, pred[0, 4, 0, 0], gt[0, 4, 0, 0])

        optim.step()



    gt[0, 4, 0, 0]
    pred[0, 4, 0, 0]

    mse = nn.MSELoss(reduction="sum")
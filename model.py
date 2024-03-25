# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/yolo_v1.py
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from eval import get_iou

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
    
        self.conv1_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3) # "7×7×64-s-2"
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv2_1 = ConvBlock(64, 192, kernel_size=3, padding=1) # "3×3×192"
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv3_1 = ConvBlock(192, 128, kernel_size=1) # "1×1×128"
        self.conv3_2 = ConvBlock(128, 256, kernel_size=3, padding=1)  # "3×3×256"
        self.conv3_3 = ConvBlock(256, 256, kernel_size=1) # "1×1×256"
        self.conv3_4 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv4_1 = ConvBlock(512, 256, kernel_size=1) # "1×1×256"
        self.conv4_2 = ConvBlock(256, 512, kernel_size=3, padding=1) # "3×3×512"
        self.conv4_3 = ConvBlock(512, 512, kernel_size=1) # "1×1×512"
        self.conv4_4 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # "2×2×s-2"

        self.conv5_1 = ConvBlock(1024, 512, kernel_size=1) # "1×1×512"
        self.conv5_2 = ConvBlock(512, 1024, kernel_size=3, padding=1) # "3×3×1024"


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool3(x)

        for _ in range(4): # "×4"
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool4(x)

        for _ in range(2): # "×2"
            x = self.conv5_1(x)
            x = self.conv5_2(x)
        return x


class YOLOv1(nn.Module):
    def __init__(
            self,
            img_size=448,
            n_cells=7,
            n_bboxes=2,
            n_classes=20,
            lamb_coord=5,
            lamb_noobj=0.5,
        ):
        super().__init__()

        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_classes = n_classes
        self.lamb_coord = lamb_coord
        self.lamb_noobj = lamb_noobj

        self.cell_size = img_size // n_cells

        self.darknet = Darknet()

        self.conv5_3 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv5_4 = ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1) # "3×3×1024-s-2"

        self.conv6_1 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"
        self.conv6_2 = ConvBlock(1024, 1024, kernel_size=3, padding=1) # "3×3×1024"

        self.proj1 = nn.Linear(1024 * n_cells * n_cells, 4096)
        self.proj2 = nn.Linear(
            4096, n_cells * n_cells * (5 * n_bboxes + n_classes)
        )
        # "A dropout layer with rate = .5 after the first connected layer prevents co-adaptation
        # between layers"
        self.drop = nn.Dropout(0.5)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.darknet(x)

        x = self.conv5_3(x)
        x = self.conv5_4(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
    
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.proj1(x)
        x = self.drop(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE)
        x = self.proj2(x)
        x = torch.sigmoid(x)
        # "We use a linear activation function for the final layer and all other layers use
        # the leaky rectified linear activation."

        x = x.view(
            (-1, (5 * self.n_bboxes + self.n_classes), self.n_cells, self.n_cells),
        )
        return x

    # def get_obj_mask(coord_gt, cls_gt):
    #     cell_size = 64
    #     batch_size = cls_gt.size(0)
    #     x_idx = ((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2 // cell_size).long()
    #     y_idx = ((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2 // cell_size).long()
    #     appears = torch.zeros(size=(batch_size, 7, 7), dtype=torch.bool)
    #     for batch_idx in range(batch_size):
    #         for bbox_idx in range(cls_gt.size(1)):
    #             cls_idx = cls_gt[batch_idx, bbox_idx].item()
    #             if cls_idx != 20:
    #                 appears[batch_idx, x_idx[batch_idx, bbox_idx], y_idx[batch_idx, bbox_idx]] = True
    #     return appears

    def get_obj_mask(coord_gt, cls_gt):
        n_cells=7
        cell_size = 64
        batch_size = cls_gt.size(0)
        x_idx = ((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2 // cell_size).long()
        y_idx = ((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2 // cell_size).long()
        
        appears = torch.zeros(size=(batch_size, (n_cells ** 2) * 2), dtype=torch.bool)
        idx = (x_idx * n_cells + y_idx)
        appears[
            torch.arange(batch_size).repeat_interleave(cls_gt.size(1))[(cls_gt != 20).view(-1)],
            idx.view(-1)[(cls_gt != 20).view(-1)],
        ] = 1
        appears[
            torch.arange(batch_size).repeat_interleave(cls_gt.size(1))[(cls_gt != 20).view(-1)],
            idx.view(-1)[(cls_gt != 20).view(-1)] + n_cells ** 2,
        ] = 1
        return appears

    # def _encode(self, coord_gt, cls_gt):
    #     img_size=448
    #     # a = (((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2) % cell_size) / cell_size
    #     # b = (((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2) % cell_size) / cell_size
    #     # c = (coord_gt[:, :, 2] - coord_gt[:, :, 0]) / img_size
    #     # d = (coord_gt[:, :, 3] - coord_gt[:, :, 1]) / img_size
    #     # torch.cat([a, b, c, d], dim=2)

    def _encode(self, coord_gt, cls_gt):
        cell_size = 64
        img_size=448
        obj_mask = get_obj_mask(coord_gt, cls_gt)
        pred_coord[obj_mask].shape
        obj_mask
        coord_gt
        cls_gt
        obj_ms


        
        # "We parametrize the bounding box x and y coordinates to be offsets
        # of a particular grid cell location so they are also bounded between 0 and 1."
        gt["x"] = gt.apply(
            lambda x: (((x["l"] + x["r"]) / 2) % self.cell_size) / self.cell_size,
            axis=1
        )
        gt["y"] = gt.apply(
            lambda x: (((x["t"] + x["b"]) / 2) % self.cell_size) / self.cell_size,
            axis=1
        )
        # "We normalize the bounding box width and height by the image width and height
        # so that they fall between 0 and 1."
        gt["w"] = gt.apply(lambda x: (x["r"] - x["l"]) / self.img_size, axis=1)
        gt["h"] = gt.apply(lambda x: (x["b"] - x["t"]) / self.img_size, axis=1)

        gt["x_grid"] = gt.apply(
            lambda x: int((x["l"] + x["r"]) / 2 / self.cell_size), axis=1
        )
        gt["y_grid"] = gt.apply(
            lambda x: int((x["t"] + x["b"]) / 2 / self.cell_size), axis=1
        )
        return gt

    def decode(self, x):
        # x = out
        bbox = x.clone()

        bbox[:, (2, 7), ...] *= self.img_size # w
        bbox[:, (3, 8), ...] *= self.img_size # h
        bbox[:, (0, 5), ...] *= self.cell_size # x
        bbox[:, (0, 5), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(0) # x
        bbox[:, (1, 6), ...] *= self.cell_size # y
        bbox[:, (1, 6), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(1) # y

        l = bbox[:, (0, 5), ...] - bbox[:, (2, 7), ...] / 2
        t = bbox[:, (1, 6), ...] - bbox[:, (3, 8), ...] / 2
        r = bbox[:, (0, 5), ...] + bbox[:, (2, 7), ...] / 2
        b = bbox[:, (1, 6), ...] + bbox[:, (3, 8), ...] / 2

        bbox[:, (0, 5), ...] = l
        bbox[:, (1, 6), ...] = t
        bbox[:, (2, 7), ...] = r
        bbox[:, (3, 8), ...] = b

        bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...] = torch.clip(
            bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...], min=0, max=self.img_size
        )

        bbox1 = torch.cat([bbox[:, : 5, ...], bbox[:, 10:, ...]], dim=1)
        bbox1 = rearrange(bbox1, pattern="b c h w -> b (h w) c")

        bbox2 = torch.cat([bbox[:, 5: 10, ...], bbox[:, 10:, ...]], dim=1)
        bbox2 = rearrange(bbox2, pattern="b c h w -> b (h w) c")

        bbox = torch.cat([bbox1, bbox2], dim=1)
        # return torch.cat([bbox[..., : 5], bbox[..., 5:]], dim=2)
        return bbox[..., : 4], bbox[..., 4], bbox[..., 5:]


    def denormalize_xywh(norm_xywh):
        # norm_xywh = norm_pred_xywh
        xywh = norm_xywh.clone()
        xywh[:, :, :, 0] *= cell_size
        xywh[:, :, :, 1] *= cell_size
        xywh[:, :, :, 2] *= img_size
        xywh[:, :, :, 3] *= img_size
        return xywh


    def xywh_to_ltrb(xywh):
        l = torch.clip(xywh[:, :, :, 0] - xywh[:, :, :, 2] / 2, min=0)
        t = torch.clip(xywh[:, :, :, 1] - xywh[:, :, :, 3] / 2, min=0)
        r = torch.clip(xywh[:, :, :, 0] + xywh[:, :, :, 2] / 2, max=img_size)
        b = torch.clip(xywh[:, :, :, 1] + xywh[:, :, :, 3] / 2, max=img_size)
        return torch.stack([l, t, r, b], dim=3)
    

    def get_loss(self, x, norm_gt_xywh, gt_cls_idx, obj_mask, noobj_mask):
        """
        norm_gt_xywh: [B, N, K, 4]
        gt_cls_idx: [B, N, K, 1]
            L: n_bboxes (2)
            K: n_bboxes_per_cell (1)
        """
        img_size=448
        cell_size=448//7
        n_cells=7
        n_bboxes=2

        out = self(x)
        # out.min(), out.max()

        batch_size = out.size(0)
        # x1, x2, y1, y2, w1, w2, h1, h2, conf1, conf2
        # norm_pred_xywh1 = rearrange(out[:, 0: 8: n_bboxes], pattern="b (n c) h w -> b (h w) c n", n=4)
        # norm_pred_xywh2 = rearrange(out[:, 1: 8: n_bboxes], pattern="b (n c) h w -> b (h w) c n", n=4)
        # norm_pred_xywh = torch.cat([norm_pred_xywh1, norm_pred_xywh2], dim=2)
        norm_pred_xywh = rearrange(out[:, 0: 8], pattern="b (n c) h w -> b (h w) c n", n=4)
        pred_xywh = denormalize_xywh(norm_pred_xywh)
        pred_ltrb = xywh_to_ltrb(pred_xywh)

        gt_xywh = denormalize_xywh(norm_gt_xywh)
        gt_ltrb = xywh_to_ltrb(gt_xywh)

        iou = get_iou(pred_ltrb, gt_ltrb)
        max_iou, idx_max = torch.max(iou, dim=2, keepdim=False)
        iou_mask = F.one_hot(idx_max[:, :, 0], num_classes=n_bboxes)[:, :, :, None].repeat(1, 1, 1, 4).bool()
        resp_mask = iou_mask * obj_mask[:, :, None, None].repeat(1, 1, n_bboxes, 4)
        resp_norm_pred_xywh = resp_mask * norm_pred_xywh
        





        norm_pred_xywh.shape, norm_gt_xywh.shape
        # iou.shape
        iou.shape
        idx_max.shape



        norm_pred_xywh.shape
        norm_pred_xywh.shape


        pred_conf1 = out[:, 8: 9].view(batch_size, 1, -1).permute(0, 2, 1)
        pred_conf2 = out[:, 9: 10].view(batch_size, 1, -1).permute(0, 2, 1)

        pred_cls_logit = out[:, 10: 30].view(batch_size, 20, -1).permute(0, 2, 1)


        iou.shape




        obj_mask = get_obj_mask(coord_gt, cls_gt)
        


        # conf_order = torch.argsort(pred_conf[..., None].repeat(1, 1, 4), dim=1)
        conf_order = torch.argsort(pred_conf, dim=1)
        sorted_pred_coord = torch.gather(pred_coord, dim=1, index=conf_order[..., None].repeat(1, 1, 4))
        # pred_coord.shape, sorted_pred_coord.shape
        for batch_idx in range(4):
            batch_idx=2
            iou = get_iou(coord_gt[batch_idx], sorted_pred_coord[batch_idx])
            iou.shape
        




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
        xy_loss = self.lamb_coord * F.mse_loss(pred_xy_obj, gt_xy_obj, reduction="mean")

        pred_wh_obj = pred[..., wh_indices][obj_indices]
        gt_wh_obj = gt[..., wh_indices][obj_indices]
        wh_loss = self.lamb_coord * F.mse_loss(pred_wh_obj ** 0.5, gt_wh_obj ** 0.5)
        coord_loss = xy_loss + wh_loss

        ### Confidence loss
        pred_conf_obj = pred[..., conf_indices][obj_indices]
        gt_conf_obj = gt[..., conf_indices][obj_indices]
        conf_loss_obj = F.mse_loss(pred_conf_obj, gt_conf_obj)

        pred_conf_noobj = pred[..., conf_indices][noobj_indices]
        gt_conf_noobj = gt[..., conf_indices][noobj_indices]
        conf_loss_noobj = self.lamb_noobj * F.mse_loss(pred_conf_noobj, gt_conf_noobj)

        conf_loss = conf_loss_obj + conf_loss_noobj

        ### Classification loss
        pred_cls_obj = pred[..., cls_indices][obj_indices]
        gt_cls_obj = gt[..., cls_indices][obj_indices]
        cls_loss = F.mse_loss(pred_cls_obj, gt_cls_obj)

        loss = coord_loss + conf_loss + cls_loss
        loss /= b
        return loss


if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn((4, 3, 448, 448))
    out = model(x)

    out.shape

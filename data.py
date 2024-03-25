# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/voc.py

# "We train the network on the training and validation data sets from PASCAL VOC 2007 and 2012.
# When testing on 2012 we also include the VOC 2007 test data for training."

from pathlib import Path
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as et
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
from collections import defaultdict

# import config
from utils import VOC_CLASSES
# draw_bboxes, get_image_dataset_mean_and_std


class VOC2012Dataset(Dataset):
    def __init__(
        self,
        annot_dir,
        augment=True,
        img_size=448,
        n_cells=7,
        n_bboxes=2,
        n_bboxes_per_cell=1,
    ):
        super().__init__()

        self.augment = augment
        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_bboxes_per_cell = n_bboxes_per_cell

        self.cell_size = img_size // n_cells

        self.annots = list(Path(annot_dir).glob("*.xml"))

    def __len__(self):
        return len(self.annots)

    @staticmethod
    def get_image_from_xml_path(xml_path):
        # xml_path = "/home/dmeta0304/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2012_004295.xml"
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
        return Image.open(img_path).convert("RGB")

    def parse_xml_file(self, xml_path):
        image = self.get_image_from_xml_path(xml_path)
        # xml_path = "/home/dmeta0304/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2012_004272.xml"
        # cell_size = 64
        # n_bboxes_per_cell=2

        # for bbox in xroot.findall("object"):
        #     l = int(bbox.find("bndbox").find("xmin").text)
        #     t = int(bbox.find("bndbox").find("ymin").text)
        #     r = int(bbox.find("bndbox").find("xmax").text)
        #     b = int(bbox.find("bndbox").find("ymax").text)

        #     x = (l + r) // 2
        #     y = (t + b) // 2
        #     w = r - l
        #     h = b - t

        #     x_cell_idx = x // cell_size
        #     y_cell_idx = y // cell_size
        #     x_cell_idx, y_cell_idx


        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        gt_ltrb = torch.tensor(
            [
                (
                    round(float(bbox.find("bndbox").find("xmin").text)),
                    round(float(bbox.find("bndbox").find("ymin").text)),
                    round(float(bbox.find("bndbox").find("xmax").text)),
                    round(float(bbox.find("bndbox").find("ymax").text)),
                ) for bbox in xroot.findall("object")
            ],
            dtype=torch.int32,
        )
        gt_cls_idx = torch.tensor(
            [
                (
                    VOC_CLASSES.index(xroot.find("object").find("name").text),
                )
                for _
                in xroot.findall("object")
            ],
            dtype=torch.int32,
        )
        return image, gt_ltrb, gt_cls_idx

    def _randomly_flip_horizontally(self, image, gt_ltrb, p=0.5):
        w, _ = image.size
        if random.random() > 1 - p:
            image = TF.hflip(image)
            gt_ltrb[:, 0] = w - gt_ltrb[:, 0]
            gt_ltrb[:, 2] = w - gt_ltrb[:, 2]
        return image, gt_ltrb

    # "We introduce random scaling and translations of up to 20% of the original image size."
    def _randomly_scale(self, image, gt_ltrb, transform_ratio=0.2):
        w, h = image.size
        scale = random.uniform(1 - transform_ratio, 1 + transform_ratio)
        image = TF.resize(image, size=(round(h * scale), round(w * scale)), antialias=True)

        torch.round_(gt_ltrb * scale)
        return image, gt_ltrb

    # def _randomly_shift(self, image, gt_ltrb, ori_w, ori_h, transform_ratio=0.2):
    def _randomly_shift(self, image, gt_ltrb, transform_ratio=0.2):
        ori_w, ori_h = image.size
        dx = round(ori_w * random.uniform(-transform_ratio, transform_ratio))
        dy = round(ori_h * random.uniform(-transform_ratio, transform_ratio))

        image = TF.pad(image, padding=(dx, dy, -dx, -dy), padding_mode="constant")

        gt_ltrb[:, 0] += dx
        gt_ltrb[:, 1] += dy
        gt_ltrb[:, 2] += dx
        gt_ltrb[:, 3] += dy

        w, h = image.size
        gt_ltrb[:, (0, 2)].clip_(0, w)
        gt_ltrb[:, (0, 2)].clip_(0, h)
        return image, gt_ltrb

    def _randomly_adjust_b_and_s(self, image):
        # We also randomly adjust the exposure and saturation of the image by up to a factor
        # of 1.5 in the HSV color space."
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _crop_center(self, image, gt_ltrb):
        w, h = image.size

        image = TF.center_crop(image, output_size=self.img_size)

        gt_ltrb[:, (0, 2)] += (self.img_size - w) // 2 
        gt_ltrb[:, (1, 3)] += (self.img_size - h) // 2 
        return image, gt_ltrb

    @staticmethod
    def ltrb_to_xywh(ltrb):
        return torch.stack(
            [
                (ltrb[:, 0] + ltrb[:, 2]) // 2,
                (ltrb[:, 1] + ltrb[:, 3]) // 2,
                ltrb[:, 2] - ltrb[:, 0],
                ltrb[:, 3] - ltrb[:, 1],
            ],
            dim=1,
        )

    def normalize_xywh(self, xywh):
        norm_xywh = xywh.clone().float()
        norm_xywh[:, 0] = norm_xywh[:, 0] % self.cell_size / self.cell_size
        norm_xywh[:, 1] = norm_xywh[:, 1] % self.cell_size / self.cell_size
        norm_xywh[:, 2] /= self.img_size
        norm_xywh[:, 3] /= self.img_size
        return norm_xywh

    def xy_to_obj_idx(self, xy):
        return (xy // self.cell_size).to(torch.long)

    def row_idx_to_obj_mask(self, row_idx):
        """
        $\mathbb{1}^{obj}_{i}$; "If object appears in cell $i$."
        """
        # obj_mask = torch.zeros(size=((self.n_cells ** 2) * self.n_bboxes,), dtype=torch.bool)
        # obj_mask[row_idx] = True
        # obj_mask[row_idx + (self.n_cells ** 2)] = True
        obj_mask = torch.zeros(size=((self.n_cells ** 2),), dtype=torch.bool)
        obj_mask[row_idx] = True
        return obj_mask

    def row_idx_to_noobj_mask(self, row_idx):
        """
        $\mathbb{1}^{noobj}_{ij}$; "The $j$th bounding box predictor in cell $i$
        is 'responsible' for that prediction."
        """
        obj_mask = self.row_idx_to_obj_mask(row_idx)
        return ~obj_mask

    def obj_idx_to_row_idx(self, obj_idx):
        row_idx = obj_idx[:, 0] * self.n_cells + obj_idx[:, 1]
        cnts = defaultdict(int)
        exists = defaultdict(bool)
        valid_indices = list()
        for idx in range(row_idx.size(0)):
            row = row_idx[idx].item()
            if cnts[row] < self.n_bboxes_per_cell:
                i = 0
                while exists[row + i * (self.n_cells ** 2)]:
                    i += 1
                cnts[row] += 1
                exists[row + i * (self.n_cells ** 2)] = True
                valid_indices.append(idx)
        dedup_row_idx = torch.tensor(list(exists.keys()), dtype=torch.long)
        return valid_indices, dedup_row_idx

    def __getitem__(self, idx):
        xml_path = self.annots[idx]
        image, gt_ltrb, gt_cls_idx = self.parse_xml_file(xml_path)

        if self.augment:
            # print(gt_ltrb.shape)
            image, gt_ltrb = self._randomly_flip_horizontally(image=image, gt_ltrb=gt_ltrb)
            # ori_w, ori_h = image.size
            image, gt_ltrb = self._randomly_shift(image=image, gt_ltrb=gt_ltrb)
            image, gt_ltrb = self._randomly_scale(image=image, gt_ltrb=gt_ltrb)
            image = self._randomly_adjust_b_and_s(image)
            image, gt_ltrb = self._crop_center(image=image, gt_ltrb=gt_ltrb)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))
        gt_ltrb = gt_ltrb.clip_(0, self.img_size)

        gt_xywh = self.ltrb_to_xywh(gt_ltrb)
        obj_idx = self.xy_to_obj_idx(gt_xywh[:, : 2])
        valid_indices, row_idx = self.obj_idx_to_row_idx(obj_idx)
        dedup_gt_xywh = gt_xywh[valid_indices]
        dedup_norm_gt_xywh = self.normalize_xywh(dedup_gt_xywh)
        dedup_gt_cls_idx = gt_cls_idx[valid_indices]

        new_norm_gt_xywh = torch.zeros(
            size=((self.n_cells ** 2) * self.n_bboxes_per_cell, 4),
            dtype=torch.float32,
        )
        new_norm_gt_xywh[row_idx] = dedup_norm_gt_xywh

        new_gt_cls_idx = torch.zeros(
            size=((self.n_cells ** 2) * self.n_bboxes_per_cell, 1),
            dtype=torch.int32,
        )
        new_gt_cls_idx[row_idx] = dedup_gt_cls_idx

        obj_mask = self.row_idx_to_obj_mask(row_idx)
        noobj_mask = self.row_idx_to_noobj_mask(row_idx)
        return image, new_norm_gt_xywh, new_gt_cls_idx, obj_mask, noobj_mask


class DynamicPadding(object):
    def __call__(self, batch):
        images = list()
        gt_ltrbs = list()
        gt_cls_idxs = list()
        max_n_objs = 0
        for image, gt_ltrb, gt_cls_idx in batch:
            images.append(image)
            gt_ltrbs.append(gt_ltrb)
            gt_cls_idxs.append(gt_cls_idx)

            n_objs = gt_ltrb.size(0)
            if n_objs > max_n_objs:
                max_n_objs = n_objs

        image = torch.stack(images)
        gt_ltrb = torch.stack(
            [
                torch.cat(
                    [
                        gt_ltrb,
                        # torch.full(
                        #     size=(max_n_objs - gt_ltrb.size(0), 4),
                        #     fill_value=len(VOC_CLASSES),
                        #     dtype=torch.int32,
                        # )
                        torch.zeros(
                            size=(max_n_objs - gt_ltrb.size(0), 4), dtype=torch.int32,
                        ),
                    ],
                    dim=0,
                )
                for gt_ltrb
                in gt_ltrbs
            ]
        )
        gt_cls_idx = torch.stack(
            [
                torch.cat(
                    [
                        gt_cls_idx,
                        torch.full(
                            size=(max_n_objs - gt_cls_idx.size(0),),
                            fill_value=len(VOC_CLASSES),
                            dtype=torch.int32,
                        )
                    ],
                    dim=0,
                )
                for gt_cls_idx
                in gt_cls_idxs
            ]
        )
        return image, gt_ltrb, gt_cls_idx


# def obj_idx_to_obj_mask(obj_idx):
#     mask = torch.zeros(size=(7, 7), dtype=torch.bool)
#     mask[obj_idx[:, 0], obj_idx[:, 1]] = True
#     return mask


# def obj_idx_to_noobj_mask(obj_idx):
#     obj_mask = obj_idx_to_obj_mask(obj_idx)
#     return ~obj_mask[None, ...].repeat(2, 1, 1)

# def get_dedup_row_idx(row_idx, n_bboxes_per_cell=1):
#     cnts = defaultdict(int)
#     valid_indices = list()
#     for idx in range(row_idx.size(0)):
#         row = row_idx[idx].item()
#         if cnts[row] < n_bboxes_per_cell:
#             valid_indices.append(idx)
#         cnts[row] += 1
#     return valid_indices


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/home/dmeta0304/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations", augment=True)
    dl = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=True, drop_last=True)
    di = iter(dl)
    image, norm_gt_xywh, gt_cls_idx, obj_mask, noobj_mask = next(di)
    # norm_gt_xywh = norm_gt_xywh.permute(0, 2, 1)
    # gt_cls_idx = gt_cls_idx.permute(0, 2, 1)
    norm_gt_xywh = norm_gt_xywh[:, :, None, :]
    gt_cls_idx = gt_cls_idx[:, :, None, :]
    norm_gt_xywh.shape, gt_cls_idx.shape, obj_mask.shape, noobj_mask.shape


    # cell_size = 64
    # n_cells= 7
    # n_bboxes_per_cell = 1
    # for i in range(10000):
    #     image, gt_ltrb, gt_cls_idx = ds[i]
    #     gt_xywh = ltrb_to_xywh(gt_ltrb)
    #     obj_idx = xy_to_obj_idx(gt_xywh[:, : 2])
    #     if obj_idx.size(0) != torch.unique(obj_idx, dim=0).size(0):
    #         break

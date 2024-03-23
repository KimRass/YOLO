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
    def __init__(self, annot_dir, img_size=448, n_cells=7, n_bboxes=2, augment=True):
        super().__init__()

        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.augment = augment

        self.cell_size = img_size // n_cells

        self.annots = list(Path(annot_dir).glob("*.xml"))

    def _randomly_flip_horizontally(self, image, coord_gt, p=0.5):
        w, _ = image.size
        if random.random() > 1 - p:
            image = TF.hflip(image)
            coord_gt[:, 0] = w - coord_gt[:, 0]
            coord_gt[:, 2] = w - coord_gt[:, 2]
        return image, coord_gt

    # "We introduce random scaling and translations of up to 20% of the original image size."
    def _randomly_scale(self, image, coord_gt, transform_ratio=0.2):
        w, h = image.size
        scale = random.uniform(1 - transform_ratio, 1 + transform_ratio)
        image = TF.resize(image, size=(round(h * scale), round(w * scale)), antialias=True)

        torch.round_(coord_gt * scale)
        return image, coord_gt

    def _randomly_shift(self, image, coord_gt, ori_w, ori_h, transform_ratio=0.2):
        dx = round(ori_w * random.uniform(-transform_ratio, transform_ratio))
        dy = round(ori_h * random.uniform(-transform_ratio, transform_ratio))

        image = TF.pad(image, padding=(dx, dy, -dx, -dy), padding_mode="constant")

        coord_gt[:, 0] += dx
        coord_gt[:, 1] += dy
        coord_gt[:, 2] += dx
        coord_gt[:, 3] += dy

        w, h = image.size
        coord_gt[:, (0, 2)].clip_(0, w)
        coord_gt[:, (0, 2)].clip_(0, h)
        return image, coord_gt

    def _randomly_adjust_b_and_s(self, image):
        # We also randomly adjust the exposure and saturation of the image by up to a factor
        # of 1.5 in the HSV color space."
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _crop_center(self, image, coord_gt):
        w, h = image.size

        image = TF.center_crop(image, output_size=self.img_size)

        coord_gt[:, (0, 2)] += (self.img_size - w) // 2 
        coord_gt[:, (1, 3)] += (self.img_size - h) // 2 
        return image, coord_gt

    def __len__(self):
        return len(self.annots)

    @staticmethod
    def parse_xml_file(xml_path):
        # xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000027.xml"
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()

        img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
        image = Image.open(img_path).convert("RGB")

        coord_gt = torch.tensor(
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
        cls_gt = torch.tensor(
            [
                VOC_CLASSES.index(xroot.find("object").find("name").text)
                for _
                in xroot.findall("object")
            ],
            dtype=torch.int32,
        )
        return image, coord_gt, cls_gt

    def __getitem__(self, idx):
        xml_path = self.annots[idx]
        image, coord_gt, cls_gt = self.parse_xml_file(xml_path)
        ori_w, ori_h = image.size

        if self.augment:
            # print(coord_gt.shape)
            image, coord_gt = self._randomly_flip_horizontally(image=image, coord_gt=coord_gt)
            image, coord_gt = self._randomly_shift(
                image=image, coord_gt=coord_gt, ori_w=ori_w, ori_h=ori_h
            )
            image, coord_gt = self._randomly_scale(image=image, coord_gt=coord_gt)
            image = self._randomly_adjust_b_and_s(image)
            image, coord_gt = self._crop_center(image=image, coord_gt=coord_gt)
        coord_gt = coord_gt.clip_(0, self.img_size)

        image = TF.to_tensor(image)
        # get_image_dataset_mean_and_std
        image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))
        return image, coord_gt, cls_gt
        # gt = self._encode(gt)
        # return image, gt


class DynamicPadding(object):
    def __call__(self, batch):
        images = list()
        coord_gts = list()
        cls_gts = list()
        max_n_objs = 0
        for image, coord_gt, cls_gt in batch:
            images.append(image)
            coord_gts.append(coord_gt)
            cls_gts.append(cls_gt)

            n_objs = coord_gt.size(0)
            if n_objs > max_n_objs:
                max_n_objs = n_objs

        image = torch.stack(images)
        coord_gt = torch.stack(
            [
                torch.cat(
                    [
                        coord_gt,
                        # torch.full(
                        #     size=(max_n_objs - coord_gt.size(0), 4),
                        #     fill_value=len(VOC_CLASSES),
                        #     dtype=torch.int32,
                        # )
                        torch.zeros(
                            size=(max_n_objs - coord_gt.size(0), 4), dtype=torch.int32,
                        ),
                    ],
                    dim=0,
                )
                for coord_gt
                in coord_gts
            ]
        )
        cls_gt = torch.stack(
            [
                torch.cat(
                    [
                        cls_gt,
                        torch.full(
                            size=(max_n_objs - cls_gt.size(0),),
                            fill_value=len(VOC_CLASSES),
                            dtype=torch.int32,
                        )
                    ],
                    dim=0,
                )
                for cls_gt
                in cls_gts
            ]
        )
        return image, coord_gt, cls_gt


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


def xy_to_cell_idx(xy):
    return xy // 64


def cell_idx_to_mask(cell_idx):
    mask = torch.zeros(size=(7, 7), dtype=torch.bool)
    mask[cell_idx[:, 0], cell_idx[:, 1]] = True
    return mask


def normalize_xywh(xywh):
    img_size = 448
    norm_xywh = xywh.clone().float()
    norm_xywh[:, 0] = norm_xywh[:, 0] % cell_size / cell_size
    norm_xywh[:, 1] = norm_xywh[:, 1] % cell_size / cell_size
    norm_xywh[:, 2] /= img_size
    norm_xywh[:, 3] /= img_size
    return norm_xywh


def cell_idx_to_row_idx(cell_idx):
    return cell_idx[:, 0] * n_cells + cell_idx[:, 1]


def get_nonduplicate_indices(row_idx, n_bbox_per_cell=1):
    cnts = defaultdict(int)
    valid_indices = list()
    for idx in range(row_idx.size(0)):
        row = row_idx[idx].item()
        if cnts[row] < n_bbox_per_cell:
            valid_indices.append(idx)
        cnts[row] += 1
    return valid_indices


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations", augment=False)
    cell_size = 64
    n_cells= 7
    n_bbox_per_cell = 1
    for i in range(10000):
        image, ltrb_gt, cls_gt = ds[i]
        gt_xywh = ltrb_to_xywh(ltrb_gt)
        cell_idx = xy_to_cell_idx(gt_xywh[:, : 2])
        if cell_idx.size(0) != torch.unique(cell_idx, dim=0).size(0):
            break

    row_idx = cell_idx_to_row_idx(cell_idx)
    valid_indices = get_nonduplicate_indices(row_idx)
    dedup_row_idx = row_idx[valid_indices]
    dedup_gt_xywh = gt_xywh[valid_indices]
    dedup_norm_xywh = normalize_xywh(dedup_gt_xywh)
    dedup_cls_gt = cls_gt[valid_indices]
    # mask = cell_idx_to_mask(cell_idx)
    dedup_row_idx.size(0)
    
    row_idx_mask = torch.full(size=((n_cells ** 2) * n_bbox_per_cell,), fill_value=(n_cells ** 2) * n_bbox_per_cell)
    row_idx_mask[dedup_row_idx] = torch.arange(dedup_row_idx.size(0))
    row_idx_mask
            

    dl = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=True, drop_last=True, collate_fn=DynamicPadding())
    di = iter(dl)
    image, coord_gt, cls_gt = next(di)
    # coord_gt.shape, cls_gt.shape
    coord_gt.shape
    cls_gt

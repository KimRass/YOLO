# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/voc.py

# "We train the network on the training and validation data sets from PASCAL VOC 2007 and 2012.
# When testing on 2012 we also include the VOC 2007 test data for training."
import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/YOLO")
import torch.nn.functional as F
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
from vis import draw_grids_and_bboxes


class VOC2012Dataset(Dataset):
    def __init__(
        self,
        annot_dir,
        augment=True,
        img_size=448,
        n_cells=7,
        n_bboxes=2,
        n_bboxes_per_cell=1,
        n_classes=20,
    ):
        super().__init__()

        self.augment = augment
        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_bboxes_per_cell = n_bboxes_per_cell
        self.n_classes = n_classes

        self.cell_size = img_size // n_cells

        self.annots = list(Path(annot_dir).glob("*.xml"))

    def __len__(self):
        return len(self.annots)

    @staticmethod
    def get_image_from_xml_path(xml_path):
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
        return Image.open(img_path).convert("RGB")

    def parse_xml_file(self, xml_path):
        print(xml_path)
        image = self.get_image_from_xml_path(xml_path)

        # xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000170.xml"
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        gt_ltrb = torch.tensor(
            [
                (
                    round(float(obj.find("bndbox").find("xmin").text)),
                    round(float(obj.find("bndbox").find("ymin").text)),
                    round(float(obj.find("bndbox").find("xmax").text)),
                    round(float(obj.find("bndbox").find("ymax").text)),
                ) for obj in xroot.findall("object")
            ],
            dtype=torch.int32,
        )
        gt_cls_idx = torch.tensor(
            [
                VOC_CLASSES.index(obj.find("name").text)
                for obj
                in xroot.findall("object")
            ],
            dtype=torch.int64,
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
        ori_w, ori_h = image.size
        scale = random.uniform(1 - transform_ratio, 1 + transform_ratio)

        # image = TF.resize(
        #     image, size=(round(ori_h * scale), round(ori_ * scale)), antialias=True,
        # )
        image = image.resize((round(ori_w * scale), round(ori_h * scale)))

        gt_ltrb[:, 0] = torch.clip(gt_ltrb[:, 0] * scale, min=0, max=ori_w)
        gt_ltrb[:, 1] = torch.clip(gt_ltrb[:, 1] * scale, min=0, max=ori_h)
        gt_ltrb[:, 2] = torch.clip(gt_ltrb[:, 2] * scale, min=0, max=ori_w)
        gt_ltrb[:, 3] = torch.clip(gt_ltrb[:, 3] * scale, min=0, max=ori_h)
        return image, gt_ltrb

    def _randomly_shift(self, image, gt_ltrb, transform_ratio=0.2):
        ori_w, ori_h = image.size
        dx = round(ori_w * random.uniform(-transform_ratio, transform_ratio))
        dy = round(ori_h * random.uniform(-transform_ratio, transform_ratio))

        # image = TF.pad(image, padding=(dx, dy, -dx, -dy), padding_mode="constant")
        image = image.transform(
            size=(ori_w, ori_h),
            method=Image.AFFINE,
            data=(1, 0, -dx, 0, 1, -dy),
        )

        gt_ltrb[:, 0] = torch.clip(gt_ltrb[:, 0] + dx, min=0, max=ori_w)
        gt_ltrb[:, 1] = torch.clip(gt_ltrb[:, 1] + dy, min=0, max=ori_h)
        gt_ltrb[:, 2] = torch.clip(gt_ltrb[:, 2] + dx, min=0, max=ori_w)
        gt_ltrb[:, 3] = torch.clip(gt_ltrb[:, 3] + dy, min=0, max=ori_h)
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
        "$\mathbb{1}^{obj}_{i}$"; "If object appears in cell $i$."
        """
        obj_mask = torch.zeros(size=((self.n_cells ** 2), 1, 1), dtype=torch.bool)
        obj_mask[row_idx] = True
        return obj_mask

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
            image, gt_ltrb = self._randomly_flip_horizontally(image=image, gt_ltrb=gt_ltrb)
            image, gt_ltrb = self._randomly_scale(image=image, gt_ltrb=gt_ltrb)
            image, gt_ltrb = self._randomly_shift(image=image, gt_ltrb=gt_ltrb)
            image, gt_ltrb = self._crop_center(image=image, gt_ltrb=gt_ltrb)
            # image = self._randomly_adjust_b_and_s(image)
        # image = TF.to_tensor(image)
        # image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))
        gt_ltrb = gt_ltrb.clip_(min=0, max=self.img_size)

        gt_xywh = self.ltrb_to_xywh(gt_ltrb)
        obj_idx = self.xy_to_obj_idx(gt_xywh[:, : 2])
        valid_indices, row_idx = self.obj_idx_to_row_idx(obj_idx)
        dedup_gt_xywh = gt_xywh[valid_indices]
        dedup_gt_norm_xywh = self.normalize_xywh(dedup_gt_xywh)
        gt_cls_prob = F.one_hot(gt_cls_idx, num_classes=self.n_classes)
        dedup_gt_cls_prob = gt_cls_prob[valid_indices]

        new_gt_norm_xywh = torch.zeros(
            size=((self.n_cells ** 2) * self.n_bboxes_per_cell, 4),
            dtype=torch.float32,
        )
        new_gt_norm_xywh[row_idx] = dedup_gt_norm_xywh

        new_gt_cls_prob = torch.zeros(
            size=((self.n_cells ** 2) * self.n_bboxes_per_cell, self.n_classes),
            dtype=torch.int64,
        )
        new_gt_cls_prob[row_idx] = dedup_gt_cls_prob

        obj_mask = self.row_idx_to_obj_mask(row_idx)
        # return image, new_gt_norm_xywh, new_gt_cls_prob, obj_mask
        return image, gt_ltrb, gt_cls_idx


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations", augment=True)
    idx = 100
    for _ in range(4):
        image, gt_ltrb, gt_cls_idx = ds[500]
        # gt_cls_idx
        out = draw_grids_and_bboxes(image, gt_ltrb, gt_cls_idx)
        Image.fromarray(out).show()


    dl = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=True, drop_last=True)
    di = iter(dl)

    image, gt_norm_xywh, gt_cls_prob, obj_mask = next(di)
    gt_norm_xywh = gt_norm_xywh[:, :, None, :]
    gt_cls_prob = gt_cls_prob[:, :, None, :]
    gt_norm_xywh.shape, gt_cls_prob.shape, obj_mask.shape
    gt_norm_xywh[:, :, :, 2]
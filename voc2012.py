# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/voc.py

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xml.etree.ElementTree as et
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

import config
from utils import draw_bboxes, get_image_dataset_mean_and_std

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
IMG_SIZE = 448
N_CELLS = 7
CELL_SIZE = IMG_SIZE // N_CELLS
TRANSFORM_RATIO = 0.2


def parse_xml_file(xml_path):
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()

    bboxes = pd.DataFrame([
        (
            int(bbox.find("bndbox").find("xmin").text),
            int(bbox.find("bndbox").find("ymin").text),
            int(bbox.find("bndbox").find("xmax").text),
            int(bbox.find("bndbox").find("ymax").text),
            VOC_CLASSES.index(xroot.find("object").find("name").text)
        ) for bbox in xroot.findall("object")
    ], columns=("x1", "y1", "x2", "y2", "cls"))

    img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
    image = Image.open(img_path).convert("RGB")
    return image, bboxes


class VOC2012Dataset(Dataset):
    # def __init__(self, annot_dir, transform=None):
    def __init__(self, annot_dir):
        super().__init__()

        self.annots = list(Path(annot_dir).glob("*.xml"))
        # self.transform = transform

    def _randomly_flip_horizontally(self, image, bboxes, p=0.5):
        w, _ = image.size
        if random.random() > 1 - p:
            image = TF.hflip(image)
            bboxes["x1"], bboxes["x2"] = w - bboxes["x2"], w - bboxes["x1"]
        return image, bboxes

    # "We introduce random scaling and translations of up to 20% of the original image size."
    def _randomly_scale(self, image, bboxes):
        w, h = image.size
        scale = random.uniform(1 - TRANSFORM_RATIO, 1 + TRANSFORM_RATIO)
        image = TF.resize(image, size=(round(h * scale), round(w * scale)), antialias=True)

        bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * scale))
        bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * scale))
        bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * scale))
        bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * scale))
        return image, bboxes

    def _randomly_shift(self, image, bboxes, ori_w, ori_h):
        dx = round(ori_w * random.uniform(-TRANSFORM_RATIO, TRANSFORM_RATIO))
        dy = round(ori_h * random.uniform(-TRANSFORM_RATIO, TRANSFORM_RATIO))

        image = TF.pad(image, padding=(dx, dy, -dx, -dy), padding_mode="constant")

        bboxes["x1"] += dx
        bboxes["y1"] += dy
        bboxes["x2"] += dx
        bboxes["y2"] += dy

        w, h = image.size
        bboxes[["x1", "x2"]] = bboxes[["x1", "x2"]].clip(0, w)
        bboxes[["y1", "y2"]] = bboxes[["y1", "y2"]].clip(0, h)
        return image, bboxes

    def _randomly_adjust_b_and_s(self, image):
        # We also randomly adjust the exposure and saturation of the image by up to a factor
        # of 1.5 in the HSV color space."
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _crop_center(self, image, bboxes):
        w, h = image.size

        image = TF.center_crop(image, output_size=IMG_SIZE)

        bboxes["x1"] += (IMG_SIZE - w) // 2
        bboxes["y1"] += (IMG_SIZE - h) // 2
        bboxes["x2"] += (IMG_SIZE - w) // 2
        bboxes["y2"] += (IMG_SIZE - h) // 2
        return image, bboxes

    def _encode(self, bboxes):
        # "We parametrize the bounding box x and y coordinates to be offsets
        # of a particular grid cell location so they are also bounded between 0 and 1."
        bboxes["x"] = bboxes.apply(
            lambda x: (((x["x1"] + x["x2"]) / 2) % CELL_SIZE) / CELL_SIZE,
            axis=1
        )
        bboxes["y"] = bboxes.apply(
            lambda x: (((x["y1"] + x["y2"]) / 2) % CELL_SIZE) / CELL_SIZE,
            axis=1
        )
        # "We normalize the bounding box width and height by the image width and height
        # so that they fall between 0 and 1."
        bboxes["w"] = bboxes.apply(lambda x: (x["x2"] - x["x1"]) / IMG_SIZE, axis=1)
        bboxes["h"] = bboxes.apply(lambda x: (x["y2"] - x["y1"]) / IMG_SIZE, axis=1)

        bboxes["x_grid"] = bboxes.apply(
            lambda x: int((x["x1"] + x["x2"]) / 2 / CELL_SIZE), axis=1
        )
        bboxes["y_grid"] = bboxes.apply(
            lambda x: int((x["y1"] + x["y2"]) / 2 / CELL_SIZE), axis=1
        )

        # gt = torch.zeros((25, N_CELLS, N_CELLS), dtype=torch.float)
        gt = torch.zeros((30, N_CELLS, N_CELLS), dtype=torch.float)
        for row in bboxes.itertuples():
            gt[0, row.y_grid, row.x_grid] = row.x
            gt[1, row.y_grid, row.x_grid] = row.y
            gt[2, row.y_grid, row.x_grid] = row.w
            gt[3, row.y_grid, row.x_grid] = row.h
            gt[4, row.y_grid, row.x_grid] = 1
            gt[5, row.y_grid, row.x_grid] = row.x
            gt[6, row.y_grid, row.x_grid] = row.y
            gt[7, row.y_grid, row.x_grid] = row.w
            gt[8, row.y_grid, row.x_grid] = row.h
            gt[9, row.y_grid, row.x_grid] = 1
            # gt[5 + row.cls, row.y_grid, row.x_grid] = 1
            gt[10 + row.cls, row.y_grid, row.x_grid] = 1
        return gt

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        xml_path = self.annots[idx]
        image, bboxes = parse_xml_file(xml_path)
        ori_w, ori_h = image.size

        image, bboxes = self._randomly_flip_horizontally(image=image, bboxes=bboxes)
        image, bboxes = self._randomly_shift(
            image=image, bboxes=bboxes, ori_w=ori_w, ori_h=ori_h
        )
        image, bboxes = self._randomly_scale(image=image, bboxes=bboxes)
        image = self._randomly_adjust_b_and_s(image)
        image, bboxes = self._crop_center(image=image, bboxes=bboxes)

        bboxes[["x1", "y1", "x2", "y2"]] = bboxes[["x1", "y1", "x2", "y2"]].clip(0, IMG_SIZE)
        bboxes = bboxes[(bboxes["x1"] != bboxes["x2"]) & (bboxes["y1"] != bboxes["y2"])]
        # return image, bboxes

        # get_image_dataset_mean_and_std
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))
        gt = self._encode(bboxes)
        return image, gt


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
    image, gt = ds[10]
    gt[0]
    # vis = draw_bboxes(image=image, bboxes=bboxes, grids=True)
    # vis.show()

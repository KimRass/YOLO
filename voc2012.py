# References:
    # https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit#slide=id.g15092aa245_0_319
    # https://wolfy.tistory.com/259

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
import extcolors
import random

from utils import (
    _to_array,
    _to_pil,
    load_image,
    _get_width_and_height,
    show_image
)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


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
    ], columns=("x1", "y1", "x2", "y2", "label"))

    img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
    image = Image.open(img_path).convert("RGB")
    return image, bboxes


def _normalize_bboxes_coordinates(bboxes, img, img_size=448):
    copied = bboxes.copy()

    img_w, img_h = _get_width_and_height(img)
    if img_w >= img_h:
        resize_ratio = img_size / img_w
        new_h = round(img_h * resize_ratio)
        pad = (img_size - new_h) // 2

        copied["x1"] = copied["x1"].apply(lambda x: round(x * resize_ratio))
        copied["x2"] = copied["x2"].apply(lambda x: round(x * resize_ratio))
        copied["y1"] = copied["y1"].apply(lambda x: round(x * resize_ratio + pad))
        copied["y2"] = copied["y2"].apply(lambda x: round(x * resize_ratio + pad))
    else:
        resize_ratio = img_size / img_h
        new_w = round(img_w * resize_ratio)
        pad = (img_size - new_w) // 2

        copied["x1"] = copied["x1"].apply(lambda x: round(x * resize_ratio + pad))
        copied["x2"] = copied["x2"].apply(lambda x: round(x * resize_ratio + pad))
        copied["y1"] = copied["y1"].apply(lambda x: round(x * resize_ratio))
        copied["y2"] = copied["y2"].apply(lambda x: round(x * resize_ratio))
    return copied


def generate_ground_truth(img, bboxes, img_size=448, n_cells=7):
    copied = bboxes.copy()
    ### Normalize
    copied = _normalize_bboxes_coordinates(bboxes=copied, img=img)

    cell_size = img_size // n_cells
    copied["x"] = copied.apply(
        lambda x: (((x["x1"] + x["x2"]) / 2) % cell_size) / cell_size,
        axis=1
    )
    copied["y"] = copied.apply(
        lambda x: (((x["y1"] + x["y2"]) / 2) % cell_size) / cell_size,
        axis=1
    )
    copied["w"] = copied.apply(lambda x: (x["x2"] - x["x1"]) / img_size, axis=1)
    copied["h"] = copied.apply(lambda x: (x["y2"] - x["y1"]) / img_size, axis=1)

    copied["c"] = 1
    copied["x_grid"] = copied.apply(
        lambda x: int((x["x1"] + x["x2"]) / 2 / cell_size), axis=1
    )
    copied["y_grid"] = copied.apply(
        lambda x: int((x["y1"] + x["y2"]) / 2 / cell_size), axis=1
    )

    gt = torch.zeros((30, n_cells, n_cells), dtype=torch.float64)
    for tup in copied.itertuples():
        _, _, _, _, _, obj, x, y, w, h, c, x_grid, y_grid = tup

        gt[(0, 5), y_grid, x_grid] = x
        gt[(1, 6), y_grid, x_grid] = y
        gt[(2, 7), y_grid, x_grid] = w
        gt[(3, 8), y_grid, x_grid] = h
        gt[(4, 9), y_grid, x_grid] = c
        gt[9 + obj, y_grid, x_grid] = 1
    return gt


class Transform(object):
    def __call__(self, image):
        h, w = image.size
        transform = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(max(h, w)),
                T.Resize(448, antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        x = transform(image)
        return x


class VOC2012Dataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

    def __len__(self):
        return len(list(Path(self.root).glob("*.xml")))

    def __getitem__(self, idx):
        xml_path = list(Path(self.root).glob("*.xml"))[idx]
        xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000032.xml"
        image, bboxes = parse_xml_file(xml_path)
        drawn = draw_bboxes(image)

        w, h = image.size
        scale = random.uniform(0.8, 1.2)
        T.Resize(size=(round(h * scale), round(w * scale)), antialias=True)(image).size
        bboxes[["x1", "y1", "x2", "y2"]] = bboxes[["x1", "y1", "x2", "y2"]]

        bboxes[["x1", "y1", "x2", "y2"]].apply(lambda x: list(map(int, round(x * scale))), axis=1)






        
        xroot.find("filename").text
        Image.open(
        img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text)
        temp = _normalize_bboxes_coordinates(bboxes=bboxes, img=img)
        bboxes
        temp
        gt = generate_ground_truth(img=img, bboxes=bboxes)
        image = _to_pil(img)
        if self.transform is not None:
            image = self.transform(image)
        return image, gt


if __name__ == "__main__":
    transform = Transform()
    ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=pad_to_bboxes)
    for batch, (image, gt) in enumerate(dl, start=1):
        image.shape, gt.shape

        grid = batched_image_to_grid(image=image, n_cols=4, normalize=True)
        show_image(grid)

        
        darknet = Darknet()
        yolo = YOLO(darknet=darknet, n_classes=20)
        pred = yolo(image)

# "we introduce random scaling and translations of up to 20% of the original image size. We also randomly adjust the exposure and saturation of the image by up to a factor of 1:5 in the HSV color space."
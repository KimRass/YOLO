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
import extcolors
import random

from utils import (
    _to_array,
    _to_pil,
    load_image,
    _get_width_and_height,
    show_image,
    draw_bboxes,
)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMG_SIZE = 448

# "To avoid overfitting we use dropout and extensive data augmentation. A dropout layer with rate = .5 after the first connected layer prevents co-adaptation between layers [18]. For data augmentation we introduce random scaling and translations of up to 20% of the original image size. We also randomly adjust the exposure and saturation of the image by up to a factor of 1:5 in the HSV color space. 2.3. Inference Just like in training, predicting"


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


def _normalize_bboxes_coordinates(bboxes, img, img_size=IMG_SIZE):
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


def generate_ground_truth(img, bboxes, img_size=IMG_SIZE, n_cells=7):
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
                T.Resize(IMG_SIZE, antialias=True),
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

    # def resize(image, bboxes):
    #     w, h = image.size
    #     image = TF.center_crop(image, output_size=max(h, w))
    #     image = TF.resize(image, size=IMG_SIZE, antialias=True)
        
    #     scale = IMG_SIZE / max(w, h)
    #     pad = round((IMG_SIZE - w * scale) // 2), round((IMG_SIZE - h * scale) // 2)
    #     bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * scale + pad[0]))
    #     bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * scale + pad[1]))
    #     bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * scale + pad[0]))
    #     bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * scale + pad[1]))
    #     return image, bboxes

    def _random_scale(image, bboxes):
        h, w = image.size
        scale = random.uniform(1 - 0.2, 1 + 0.2)
        image = TF.resize(image, size=(round(w * scale), round(h * scale)), antialias=True)

        bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * scale))
        bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * scale))
        bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * scale))
        bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * scale))
        return image, bboxes

    def shift_image(image, ori_w, ori_h):
        dx = round(ori_w * random.uniform(-0.2, 0.2))
        dy = round(ori_h * random.uniform(-0.2, 0.2))

        img = np.array(image)
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy > 0:
            img[: dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, : dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        image = Image.fromarray(img)
        return image

    def _resize(image):
        w, h = image.size
        img = np.array(image)
        if w > IMG_SIZE:
            img = img[:, w // 2 - IMG_SIZE // 2: w // 2 + IMG_SIZE // 2, :]
        if h > IMG_SIZE:
            img = img[h // 2 - IMG_SIZE // 2: h // 2 + IMG_SIZE // 2, :, :]

        h, w, _ = img.shape
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        canvas[
            IMG_SIZE // 2 - h // 2: IMG_SIZE // 2 - h // 2 + h,
            IMG_SIZE // 2 - w // 2: IMG_SIZE // 2 - w // 2 + w,
            :
        ] = img
        image = Image.fromarray(canvas)
        return image

    def __len__(self):
        return len(list(Path(self.root).glob("*.xml")))

    def __getitem__(self, idx):
        xml_path = list(Path(self.root).glob("*.xml"))[idx]
        xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000032.xml"
        image, bboxes = parse_xml_file(xml_path)
        ori_w, ori_h = image.size
        image, bboxes = _random_scale(image=image, bboxes=bboxes)
        image = shift_image(image, ori_w=ori_w, ori_h=ori_h)
        image = _resize(image)
        image.show()

        
        

        print(w, h)
        print(canvas[
            IMG_SIZE // 2 - h // 2: IMG_SIZE // 2 - h // 2 + h,
            IMG_SIZE // 2 - w // 2: IMG_SIZE // 2 - w // 2 + w,
            :
        ].shape)
        canvas[
            IMG_SIZE // 2 - h // 2: IMG_SIZE // 2 - h // 2 + h,
            IMG_SIZE // 2 - w // 2: IMG_SIZE // 2 - w // 2 + w,
            :
        ] = np.array(image)
        return Image.fromarray(canvas)
        w, h
        (IMG_SIZE // 2 - h, IMG_SIZE // 2 + h)


        show_image(img)
        show_image(img)
    
        center_x, center_y
        canvas
        
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        x1 = center_x - min(w, IMG_SIZE) // 2
        y1 = center_y - min(h, IMG_SIZE) // 2
        x2 = center_x + min(w, IMG_SIZE) // 2
        y2 = center_y + min(h, IMG_SIZE) // 2
        
        canvas[y1: y2, x1: x2, :] = img[y1: y2, x1: x2, :]
        w, h
        
        
        
        
        
        (center_x, center_y), (w, h)
        
        x1 = max(0, center_x - IMG_SIZE // 2)
        y1 = max(0, center_y - IMG_SIZE // 2)
        x2 = min(w, center_x + IMG_SIZE // 2)
        y2 = min(h, center_y + IMG_SIZE // 2)
        x1, y1, x2, y2
        
        x1 = max(0, center_x - w // 2)
        y1 = max(0, center_y - h // 2)
        x2 = min(w, center_x + w // 2)
        y2 = min(h, center_y + h // 2)
        # x1, y1, x2, y2 = center_x - w // 2, center_y - h // 2, center_x + w // 2, center_y + h // 2
        canvas[y1: y2, x1: x2, :] = np.array(image)
        np.array(image).shape
        h
        
        
        
        
    
        drawn = draw_bboxes(image=image, bboxes=bboxes)
        drawn.show()
        
        
        
        ratio = h / w
        image, bboxes = resize(image=image, bboxes=bboxes)


        IMG_SIZE, IMG_SIZE / ratio


        
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
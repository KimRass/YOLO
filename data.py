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

# import config
from utils import VOC_CLASSES
# draw_bboxes, get_image_dataset_mean_and_std


def parse_xml_file(xml_path):
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()

    img_path = Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text
    image = Image.open(img_path).convert("RGB")

    gt = torch.tensor(
        [
            (
                round(float(bbox.find("bndbox").find("xmin").text)),
                round(float(bbox.find("bndbox").find("ymin").text)),
                round(float(bbox.find("bndbox").find("xmax").text)),
                round(float(bbox.find("bndbox").find("ymax").text)),
                VOC_CLASSES.index(xroot.find("object").find("name").text)
            ) for bbox in xroot.findall("object")
        ],
        dtype=torch.float32,
    )
    # df_bboxes = pd.DataFrame([
    #     (
    #         round(float(bbox.find("bndbox").find("xmin").text)),
    #         round(float(bbox.find("bndbox").find("ymin").text)),
    #         round(float(bbox.find("bndbox").find("xmax").text)),
    #         round(float(bbox.find("bndbox").find("ymax").text)),
    #         VOC_CLASSES.index(xroot.find("object").find("name").text)
    #     ) for bbox in xroot.findall("object")
    # ], columns=("l", "t", "r", "b", "cls"))
    # return image, df_bboxes
    return image, gt


class VOC2012Dataset(Dataset):
    def __init__(self, annot_dir, img_size=448, n_cells=7, n_bboxes=2):
        super().__init__()

        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes

        self.cell_size = img_size // n_cells

        self.annots = list(Path(annot_dir).glob("*.xml"))

    def _randomly_flip_horizontally(self, image, df_bboxes, p=0.5):
        w, _ = image.size
        if random.random() > 1 - p:
            image = TF.hflip(image)
            df_bboxes["l"], df_bboxes["r"] = w - df_bboxes["r"], w - df_bboxes["l"]
        return image, df_bboxes

    # "We introduce random scaling and translations of up to 20% of the original image size."
    def _randomly_scale(self, image, df_bboxes, transform_ratio=0.2):
        w, h = image.size
        scale = random.uniform(1 - transform_ratio, 1 + transform_ratio)
        image = TF.resize(image, size=(round(h * scale), round(w * scale)), antialias=True)

        df_bboxes["l"] = df_bboxes["l"].apply(lambda x: round(x * scale))
        df_bboxes["t"] = df_bboxes["t"].apply(lambda x: round(x * scale))
        df_bboxes["r"] = df_bboxes["r"].apply(lambda x: round(x * scale))
        df_bboxes["b"] = df_bboxes["b"].apply(lambda x: round(x * scale))
        return image, df_bboxes

    def _randomly_shift(self, image, df_bboxes, ori_w, ori_h, transform_ratio=0.2):
        dx = round(ori_w * random.uniform(-transform_ratio, transform_ratio))
        dy = round(ori_h * random.uniform(-transform_ratio, transform_ratio))

        image = TF.pad(image, padding=(dx, dy, -dx, -dy), padding_mode="constant")

        df_bboxes["l"] += dx
        df_bboxes["t"] += dy
        df_bboxes["r"] += dx
        df_bboxes["b"] += dy

        w, h = image.size
        df_bboxes[["l", "r"]] = df_bboxes[["l", "r"]].clip(0, w)
        df_bboxes[["t", "b"]] = df_bboxes[["t", "b"]].clip(0, h)
        return image, df_bboxes

    def _randomly_adjust_b_and_s(self, image):
        # We also randomly adjust the exposure and saturation of the image by up to a factor
        # of 1.5 in the HSV color space."
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _crop_center(self, image, df_bboxes):
        w, h = image.size

        image = TF.center_crop(image, output_size=self.img_size)

        df_bboxes["l"] += (self.img_size - w) // 2
        df_bboxes["t"] += (self.img_size - h) // 2
        df_bboxes["r"] += (self.img_size - w) // 2
        df_bboxes["b"] += (self.img_size - h) // 2
        return image, df_bboxes

    def _encode(self, df_bboxes):
        if not df_bboxes.empty:
            # "We parametrize the bounding box x and y coordinates to be offsets
            # of a particular grid cell location so they are also bounded between 0 and 1."
            df_bboxes["x"] = df_bboxes.apply(
                lambda x: (((x["l"] + x["r"]) / 2) % self.cell_size) / self.cell_size,
                axis=1
            )
            df_bboxes["y"] = df_bboxes.apply(
                lambda x: (((x["t"] + x["b"]) / 2) % self.cell_size) / self.cell_size,
                axis=1
            )
            # "We normalize the bounding box width and height by the image width and height
            # so that they fall between 0 and 1."
            df_bboxes["w"] = df_bboxes.apply(lambda x: (x["r"] - x["l"]) / self.img_size, axis=1)
            df_bboxes["h"] = df_bboxes.apply(lambda x: (x["b"] - x["t"]) / self.img_size, axis=1)

            df_bboxes["x_grid"] = df_bboxes.apply(
                lambda x: int((x["l"] + x["r"]) / 2 / self.cell_size), axis=1
            )
            df_bboxes["y_grid"] = df_bboxes.apply(
                lambda x: int((x["t"] + x["b"]) / 2 / self.cell_size), axis=1
            )

        gt = torch.zeros((30, self.n_cells, self.n_cells), dtype=torch.float)
        for row in df_bboxes.itertuples():
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
            gt[10 + row.cls, row.y_grid, row.x_grid] = 1
        return gt

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        xml_path = self.annots[idx]
        image, df_bboxes = parse_xml_file(xml_path)
        ori_w, ori_h = image.size

        image, df_bboxes = self._randomly_flip_horizontally(image=image, df_bboxes=df_bboxes)
        image, df_bboxes = self._randomly_shift(
            image=image, df_bboxes=df_bboxes, ori_w=ori_w, ori_h=ori_h
        )
        image, df_bboxes = self._randomly_scale(image=image, df_bboxes=df_bboxes)
        image = self._randomly_adjust_b_and_s(image)
        image, df_bboxes = self._crop_center(image=image, df_bboxes=df_bboxes)

        image = TF.to_tensor(image)
        # get_image_dataset_mean_and_std
        image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))
        # image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        df_bboxes[["l", "t", "r", "b"]] = df_bboxes[["l", "t", "r", "b"]].clip(0, self.img_size)
        df_bboxes = df_bboxes[(df_bboxes["l"] != df_bboxes["r"]) & (df_bboxes["t"] != df_bboxes["b"])]
        return image, df_bboxes
        # gt = self._encode(df_bboxes)
        # return image, gt


if __name__ == "__main__":
    ds = VOC2012Dataset(annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations")
    dl = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=True, drop_last=True)
    di = iter(dl)
    image, gt = next(di)
    image
    gt.shape

    # xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000027.xml"
    # parse_xml_file(xml_path)
104/255
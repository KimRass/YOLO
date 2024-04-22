# References:
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/voc.py

# "We train the network on the training and validation data sets from PASCAL VOC 2007 and 2012.
# When testing on 2012 we also include the VOC 2007 test data for training."
import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/YOLO")
# sys.path.insert(0, "/home/jbkim/Desktop/workspace/YOLO")
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as et
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
from collections import defaultdict

from utils import VOC_CLASSES


class VOC2012Dataset(Dataset):
    def __init__(
        self,
        annot_dir,
        augment=True,
        mean=(0.457, 0.437, 0.404),
        std=(0.275, 0.271, 0.284),
        img_size=448,
        n_cells=7,
        n_bboxes=2,
        n_bboxes_per_cell=1,
        n_classes=20,
    ):
        super().__init__()

        self.augment = augment
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.n_cells = n_cells
        self.n_bboxes = n_bboxes
        self.n_bboxes_per_cell = n_bboxes_per_cell
        self.n_classes = n_classes
        self.cell_size = img_size // n_cells
        self.xml_paths = list(Path(annot_dir).glob("*.xml"))

    def __len__(self):
        return len(self.xml_paths)

    @staticmethod
    def get_image_from_xml_path(xml_path):
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        img_path = (
            Path(xml_path).parent.parent/"JPEGImages"
        )/xroot.find("filename").text
        return Image.open(img_path).convert("RGB")

    @staticmethod
    def get_ltrb_from_xml_path(xml_path):
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        return torch.tensor(
            [
                (
                    round(float(obj.find("bndbox").find("xmin").text)),
                    round(float(obj.find("bndbox").find("ymin").text)),
                    round(float(obj.find("bndbox").find("xmax").text)),
                    round(float(obj.find("bndbox").find("ymax").text)),
                ) for obj in xroot.findall("object")
            ],
            dtype=torch.float,
        )

    @staticmethod
    def get_label_from_xml_path(xml_path):
        xtree = et.parse(xml_path)
        xroot = xtree.getroot()
        return torch.tensor(
            [
                VOC_CLASSES.index(obj.find("name").text)
                for obj
                in xroot.findall("object")
            ],
            dtype=torch.long,
        )

    def parse_xml_file(self, xml_path):
        # xml_path = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2007_000170.xml"
        # print(xml_path)
        image = self.get_image_from_xml_path(xml_path)
        ltrb = self.get_ltrb_from_xml_path(xml_path)
        label = self.get_label_from_xml_path(xml_path)
        return image, ltrb, label

    def _flip_horizontal(self, image, ltrb, p=0.5):
        ori_w, _ = image.size
        if random.random() > 1 - p:
            image = TF.hflip(image)
            ltrb[:, 0], ltrb[:, 2] = ori_w - ltrb[:, 2], ori_w - ltrb[:, 0]
        return image, ltrb

    def _scale_randomly(self, image, ltrb, transform_ratio=0.2):
        """
        "We introduce random scaling and translations of up to 20%
        of the original image size."
        """
        scale = random.uniform(1 - transform_ratio, 1 + transform_ratio)

        ori_w, ori_h = image.size
        image = image.resize((round(ori_w * scale), round(ori_h * scale)))

        new_w, new_h = image.size
        ltrb[:, (0, 2)] = torch.clip(
            (ltrb[:, (0, 2)] * scale), min=0, max=new_w - 1,
        )
        ltrb[:, (1, 3)] = torch.clip(
            (ltrb[:, (1, 3)] * scale), min=0, max=new_h - 1,
        )
        return image, ltrb

    def _shift_randomly(self, image, ltrb, transform_ratio=0.2):
        ori_w, ori_h = image.size
        dx = round(ori_w * random.uniform(-transform_ratio, transform_ratio))
        dy = round(ori_h * random.uniform(-transform_ratio, transform_ratio))

        image = image.transform(
            size=(ori_w, ori_h),
            method=Image.AFFINE,
            data=(1, 0, -dx, 0, 1, -dy),
        )

        ltrb[:, (0, 2)] = torch.clip(ltrb[:, (0, 2)] + dx, min=0, max=ori_w - 1)
        ltrb[:, (1, 3)] = torch.clip(ltrb[:, (1, 3)] + dy, min=0, max=ori_h - 1)
        return image, ltrb

    def _crop_center(self, image, ltrb):
        ori_w, ori_h = image.size
        image = TF.center_crop(image, output_size=self.img_size)

        ltrb[:, (0, 2)] -= (ori_w - self.img_size) // 2
        ltrb[:, (1, 3)] -= (ori_h - self.img_size) // 2
        ltrb = torch.clip(ltrb, min=0, max=self.img_size - 1)
        return image, ltrb

    def _randomly_adjust_b_and_s(self, image):
        """
        "We also randomly adjust the exposure and saturation of the image
        by up to a factor of 1.5 in the HSV color space."
        """
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def transform_image_and_ltrb(self, image, ltrb):
        if self.augment:
            new_image, new_ltrb = self._flip_horizontal(image=image, ltrb=ltrb)
            new_image, new_ltrb = self._scale_randomly(
                image=new_image, ltrb=new_ltrb,
            )
            new_image, new_ltrb = self._shift_randomly(
                image=new_image, ltrb=new_ltrb,
            )
            new_image, new_ltrb = self._crop_center(
                image=new_image, ltrb=new_ltrb,
            )
            new_image = self._randomly_adjust_b_and_s(new_image)
        new_image = TF.to_tensor(new_image)
        new_image = TF.normalize(new_image, mean=self.mean, std=self.std)
        return new_image, new_ltrb

    @staticmethod
    def ltrb_to_xywh(ltrb):
        return torch.stack(
            [
                (ltrb[..., 0] + ltrb[..., 2]) / 2,
                (ltrb[..., 1] + ltrb[..., 3]) / 2,
                ltrb[..., 2] - ltrb[..., 0],
                ltrb[..., 3] - ltrb[..., 1],
            ],
            dim=-1,
        )

    def normalize_xywh(self, xywh):
        return torch.stack(
            [
                xywh[..., 0] % self.cell_size / self.cell_size,
                xywh[..., 1] % self.cell_size / self.cell_size,
                xywh[..., 2] / self.img_size,
                xywh[..., 3] / self.img_size,
            ],
            dim=-1,
        )

    def __getitem__(self, idx):
        xml_path = self.xml_paths[idx]
        image, ltrb, label = self.parse_xml_file(xml_path)
        image, ltrb = self.transform_image_and_ltrb(image=image, ltrb=ltrb)
        return image, ltrb, label
        # xywh = self.ltrb_to_xywh(ltrb)
        # norm_xywh = self.normalize_xywh(xywh)
        # return image, norm_xywh, label

    @staticmethod
    def collate_fn(batch):
        images, ltrbs, labels = list(zip(*batch))
        annots = {"ltrbs": ltrbs, "labels": labels}
        return torch.stack(images, dim=0), annots


if __name__ == "__main__":
    ds = VOC2012Dataset(
        annot_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations",
        # annot_dir="/home/jbkim/Documents/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations",
        augment=True,
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=ds.collate_fn,
    )
    for batch_idx, (image, annots) in enumerate(dl):
        if batch_idx >= 10:
            break
        # print(image.shape)
        # print([ltrb.shape for ltrb in annots["ltrbs"]])
        # print([label.shape for label in annots["labels"]])

    annots["ltrbs"][0]
    annots["labels"][0]

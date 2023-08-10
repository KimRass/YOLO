# References:
    # https://github.com/yakhyo/yolov1-pytorch
    # https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch

from torch.utils.data import DataLoader

import config
from model import YOLO
from voc2012 import VOC2012Dataset

ANNOT_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations"
BATCH_SIZE = 2
N_WORKERS = 0
ds = VOC2012Dataset(annot_dir=ANNOT_DIR)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=True, drop_last=True)

N_CLASSES = 20
model = YOLO(n_classes=N_CLASSES)

image, gt = next(iter(dl))
pred = model(image)
gt.shape, pred.shape
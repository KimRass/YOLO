# References:
    # https://github.com/yakhyo/yolov1-pytorch
    # https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from time import time
from pathlib import Path
from tqdm.auto import tqdm

import config
from utils import get_elapsed_time
from model import YOLOv1
from voc2012 import VOC2012Dataset
from loss import Yolov1Loss

torch.manual_seed(config.SEED)

print(f"""AUTOCAST = {config.AUTOCAST}""")
print(f"""N_WORKERS = {config.N_WORKERS}""")
print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")


# "For the first epochs we slowly raise the learning rate from $10^{-3}$ to $10^{-2}$.
# We continue training with $10^{-2}$ for 75 epochs, then $10^{-3}$ for 30 epochs,
# and finally $10^{-4}$ for 30 epochs."
def get_lr(step, ds_size, batch_size):
    n_steps_per_epoch = ds_size // batch_size
    if step > 0:
        lr = 1e-2 * (step - 1) / 266 + config.INIT_LR * (267 - step) / 266
    elif step > n_steps_per_epoch:
        lr = 1e-2
    elif step > 75 * n_steps_per_epoch:
        lr = 1e-3
    elif step > 105 * n_steps_per_epoch:
        lr = 1e-4
    return lr


def update_lr(lr, optim):
    optim.param_groups[0]["lr"] = lr


def save_checkpoint(epoch, step, model, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if config.N_GPUS > 1 and config.MULTI_GPU:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, str(save_path))


def load_checkpoint(model, optim, scaler):
    ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
    epoch = ckpt["epoch"]
    if config.N_GPUS > 1 and config.MULTI_GPU:
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return epoch


ds = VOC2012Dataset(annot_dir=config.ANNOT_DIR)
dl = DataLoader(
    ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.N_WORKERS,
    pin_memory=True,
    drop_last=True,
)

model = YOLOv1()
if config.N_GPUS > 0:
    model = model.to(config.DEVICE)
    if config.N_GPUS > 1 and config.MULTI_GPU:
        model = nn.DataParallel(model)

        print(f"""Using {config.N_GPUS} GPUs.""")
    else:
        print("Using single GPU.")
else:
    print("Using CPU.")

optim = SGD(
    model.parameters(),
    lr=config.INIT_LR,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY
)

scaler = GradScaler()

### Resume from checkpoint.
if config.CKPT_PATH is not None:
    init_epoch = load_checkpoint(model=model, optim=optim, scaler=scaler)
    print(f"""Resuming from epoch {init_epoch}.""")
else:
    init_epoch = 0

crit = Yolov1Loss()

ds_size = len(ds)
n_steps_per_epoch = ds_size // config.BATCH_SIZE
start_time = time()

running_loss = 0
for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
    running_loss = 0
    for step, (image, gt) in enumerate(dl, start=1):
        lr = get_lr(step=step, ds_size=ds_size, batch_size=config.BATCH_SIZE)
        update_lr(lr=lr, optim=optim)

        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        optim.zero_grad()
        with torch.autocast(
            device_type=config.DEVICE.type, dtype=torch.float16
        ) if config.AUTOCAST else nullcontext():
            pred = model(image)
            loss = crit(pred=pred, gt=gt)

        running_loss += loss.item()

        if config.AUTOCAST:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

    ### Print loss.
    if (epoch % config.N_PRINT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
        print(f"""[ {epoch}/{config.N_EPOCHS} ][ {step:,}/{n_steps_per_epoch:,} ]""", end="")
        print(F"""[ {get_elapsed_time(start_time)} ][ Loss: {running_loss / n_steps_per_epoch:.4f} ]""")

        start_time = time()

    ### Save checkpoint.
    if (epoch % config.N_CKPT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
        save_checkpoint(
            epoch=epoch,
            step=step,
            model=model,
            optim=optim,
            scaler=scaler,
            save_path=Path(__file__).parent/f"""checkpoints/{epoch}.pth""",
        )
        print(f"""Saved checkpoint at epoch {epoch}/{config.N_EPOCHS}""", end="")
        print(f""" and step {step:,}/{n_steps_per_epoch:,}.""")

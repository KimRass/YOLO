import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia
import kornia.augmentation as K


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor *= 255.0
    copied_tensor = np.clip(a=copied_tensor, a_min=0, a_max=255).astype("uint8")
    # copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor

kornia.color.hsv.rgb_to_hsv(image).shape

img = load_image("/Users/jongbeomkim/Downloads/download.png")
h, w, c = img.shape

data_aug = T.Compose(
    [
        T.RandomResizedCrop(size=(h, w), scale=(0.8, 0.8), ratio=(h / w, h / w)),
        T.ColorJitter(brightness=0.15, saturation=0.15)
    ]
)

image = T.ToTensor()(img)
temp = data_aug(image)
temp = convert_tensor_to_array(temp)
show_image(temp)

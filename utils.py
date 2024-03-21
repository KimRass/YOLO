import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from time import time
from datetime import timedelta
from einops import rearrange


# import config

# IMG_SIZE = 448
# N_CELLS = 7
# CELL_SIZE = IMG_SIZE // N_CELLS


# def draw_bboxes(image: Image.Image, bboxes: pd.DataFrame, grids: bool) -> None:
#     canvas = image.copy()
#     draw = ImageDraw.Draw(canvas)

#     if grids:
#         for i in range(1, N_CELLS): # Draw grids
#             draw.line(xy=(CELL_SIZE * i, 0, CELL_SIZE * i, IMG_SIZE), fill="rgb(150, 150, 150)", width=1)
#             draw.line(xy=(0, CELL_SIZE * i, IMG_SIZE, CELL_SIZE * i), fill="rgb(150, 150, 150)", width=1)

#     for row in bboxes.itertuples(): # Draw bboxes
#         draw.rectangle(
#             xy=(row.x1, row.y1, row.x2, row.y2),
#             outline="rgb(255, 0, 0)",
#             fill=None,
#             width=1,
#         )
#         draw.line(xy=(row.x1, row.y1, row.x2, row.y2), fill="rgb(255, 0, 0)", width=1)
#         draw.line(xy=(row.x1, row.y2, row.x2, row.y1), fill="rgb(255, 0, 0)", width=1)
#     return canvas


def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


# def draw_bboxes(image, bboxes):
#     copied = image.copy()
#     draw = ImageDraw.Draw(copied)
#     for row in bboxes.itertuples():
#         draw.rectangle(
#             xy=(row.x1, row.y1, row.x2, row.y2), outline=(0, 255, 0), width=2
#         )
#     return copied


# def load_image(img_path):
#     img_path = str(img_path)
#     img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
#     return img


# def _to_pil(img):
#     if not isinstance(img, Image.Image):
#         img = Image.fromarray(img)
#     return img


# def _to_array(img):
#     img = np.array(img)
#     return img


# def _blend_two_images(img1, img2, alpha=0.5):
#     img1 = _to_pil(img1)
#     img2 = _to_pil(img2)
#     img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
#     return _to_array(img_blended)


# def show_image(img):
#     copied_img = img.copy()
#     copied_img = _to_pil(copied_img)
#     copied_img.show()


# def save_image(img, path):
#     _to_pil(img).save(str(path))


# def _get_canvas_same_size_as_image(img, black=False):
#     if black:
#         return np.zeros_like(img).astype("uint8")
#     else:
#         return (np.ones_like(img) * 255).astype("uint8")


# def _get_width_and_height(img):
#     if img.ndim == 2:
#         h, w = img.shape
#     else:
#         h, w, _ = img.shape
#     return w, h


# def _batched_tensor_to_array(image, idx=0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     img = image.clone()[idx].permute((1, 2, 0)).detach().cpu().numpy()
#     img *= std
#     img += mean
#     img *= 255.0
#     img = np.clip(img, 0, 255).astype("uint8")
#     return img


# def draw_bboxes(image, bboxes, idx):
#     img = _batched_tensor_to_array(image=image, idx=idx)
#     canvas = _to_pil(img)
#     draw = ImageDraw.Draw(canvas)

#     for x, y, w, h, c in bboxes.CELL_SIZE * iues:
#         draw.rectangle(
#             xy=(x - w // 2, y - h // 2, x + w // 2, y + h // 2),
#             outline=(0, 0, 0),
#             fill=None,
#             width=int(c * 5)
#         )
#     canvas = _to_array(canvas)
#     return canvas


# def resize_image(img, w, h):
#     resized_img = cv2.resize(src=img, dsize=(w, h))
#     return resized_img


# def draw_bboxes(img, bboxes):
#     img_pil = _to_pil(img)
#     draw = ImageDraw.Draw(img_pil)
#     for x1, y1, x2, y2 in bboxes[["x1", "y1", "x2", "y2"]].CELL_SIZE * iues:
#         draw.rectangle(
#             xy=(x1, y1, x2, y2),
#             outline=(0, 255, 0),
#             fill=None,
#             width=2
#         )
#     return _to_array(img_pil)


# def exract_all_colors_from_segmentation_map(seg_map):
#     h, w, _ = seg_map.shape
#     colors = [
#         color
#         for color, _
#         in extcolors.extract_from_image(img=_to_pil(seg_map), tolerance=0, limit=w * h)[0]
#     ]
#     return colors


# def get_minimum_area_mask_bounding_rectangle(mask):
#     bool = (mask != 0)
#     nonzero_x = np.where(bool.any(axis=0))[0]
#     nonzero_y = np.where(bool.any(axis=1))[0]
#     if len(nonzero_x) != 0 and len(nonzero_y) != 0:
#         x1 = nonzero_x[0]
#         x2 = nonzero_x[-1]
#         y1 = nonzero_y[0]
#         y2 = nonzero_y[-1]
#     return x1, y1, x2, y2


# def get_bboxes_from_segmentation_map(seg_map):
#     colors = exract_all_colors_from_segmentation_map(seg_map)
#     ltrbs = [
#         get_minimum_area_mask_bounding_rectangle(
#             np.all(seg_map == np.array(color), axis=2)
#         )
#         for color
#         in colors
#         if color not in [(0, 0, 0), (224, 224, 192)]
#     ]
#     bboxes = pd.DataFrame(ltrbs, columns=("x1", "y1", "x2", "y2"))
#     return bboxes


# def pad_to_bboxes(batch):
#     max_n_bboxes = max([bboxes.shape[0] for _, bboxes in batch])
#     image_list, bboxes_list, = list(), list()
#     for image, bboxes in batch:
#         image_list.append(image)
#         bboxes_list.append(F.pad(bboxes, pad=(0, 0, 0, max_n_bboxes - bboxes.shape[0])))
#     return torch.stack(image_list), torch.stack(bboxes_list)
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import product
import cv2
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path

from utils import (
    VOC_CLASSES,
    to_pil,
    get_canvas_same_size_as_img,
    blend,
)


# def denormalize_array(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     copied_img = img.copy()
#     copied_img *= std
#     copied_img += mean
#     copied_img *= 255.0
#     copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
#     return copied_img


# def _tensor_to_array(tensor):
#     copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
#     copied_tensor = denormalize_array(copied_tensor)
#     return copied_tensor


# def get_class_probability_maps(pred):
#     rgb_vals = [0, 128, 255]
#     colors = tuple(product(rgb_vals, rgb_vals, rgb_vals))
#     palette = colors[0: 1] + colors[7:]
    
#     pred = pred.detach().cpu().numpy()
#     argmax = pred[:, 10:, ...].argmax(axis=1)
#     class_prob_maps = np.stack(np.vectorize(lambda x: palette[x])(argmax), axis=3).astype("uint8")
#     object_appears = np.logical_or(pred[:, 4, ...] >= 0.5, pred[:, 9, ...] >= 0.5)
#     object_appears = np.repeat(object_appears[..., None], repeats=3, axis=3)
#     class_prob_maps = class_prob_maps * object_appears
#     return class_prob_maps


# def visualize_class_probability_maps(class_prob_maps, image, idx=0):
#     img = _tensor_to_array(image[idx])
#     class_prob_map = class_prob_maps[idx]
#     resized = cv2.resize(class_prob_map, img.shape[: 2], fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
#     for k in range(7):
#         resized[64 * k - 1: 64 * k + 1, :, :] = (0, 0, 0)
#     for k in range(7):
#         resized[:, 64 * k - 1: 64 * k + 1, :] = (0, 0, 0)
#     blended = blend(img1=img, img2=resized, alpha=0.7)
#     return blended


# def draw_predicted_bboxes(pred, image, idx, img_size=448, n_cells=7):
#     is_resp = get_whether_each_predictor_is_responsible(pred)
#     pred = is_resp * pred

#     # x, y
#     cell_size = img_size // n_cells
#     pred[:, (0, 5), ...] *= cell_size
#     pred[:, (0, 5), ...] += np.indices((n_cells, n_cells), dtype="float32")[1] * cell_size
#     pred[:, (1, 6), ...] *= cell_size
#     pred[:, (1, 6), ...] += np.indices((n_cells, n_cells), dtype="float32")[0] * cell_size

#     # w, h
#     pred[:, (2, 3), ...] *= img_size
#     pred[:, (7, 8), ...] *= img_size

#     concated = np.concatenate([pred[idx, : 5, ...].reshape(5, -1), pred[idx, 5: 10, ...].reshape(5, -1)], axis=1)
#     bboxes = pd.DataFrame(concated.T, columns=["x", "y", "w", "h", "c"])
#     bboxes[["x", "y", "w", "h"]] = bboxes[["x", "y", "w", "h"]].astype("int")

#     canvas = draw_bboxes(image=image, bboxes=bboxes, idx=idx)
#     return canvas


# def batched_image_to_grid(image, n_cols, normalize=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     b, _, h, w = image.shape
#     assert b % n_cols == 0,\
#         "The batch size should be a multiple of `n_cols` argument"
#     pad = max(2, int(max(h, w) * 0.04))
#     grid = torchvision.utils.make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
#     grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

#     if normalize:
#         grid *= std
#         grid += mean
#     grid *= 255.0
#     grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

#     for k in range(n_cols + 1):
#         grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
#     for k in range(b // n_cols + 1):
#         grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
#     return grid


if __name__ == "__main__":
    xml_dir = "/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations"
    transform = Transform()
    for xml_path in Path(xml_dir).glob("*.xml"):
        # xml_path="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations/2007_000042.xml"
        try:
            img, bboxes = parse_voc2012_xml_file(xml_path)
            bboxes
        except Exception:
            continue
        bboxes = _normalize_bboxes_coordinates(bboxes=bboxes, img=img)
        img = _tensor_to_array(transform(_to_pil(img)))

        dr = draw_grids_and_bboxes(img, bboxes)
        show_image(dr)
        save_image(img=dr, path=f"""/Users/jongbeomkim/Desktop/workspace/segmentation_and_detection/yolo/ground_truths/{xml_path.stem}.jpg""")
    

    # transform = Transform()
    # ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations", transform=transform)
    # dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=pad_to_bboxes)
    # for batch, (image, gt) in enumerate(dl, start=1):
    #     b, _, _, _ = gt.shape
    #     for idx in range(b):
    #         dr = draw_predicted_bboxes(pred=gt, image=image, idx=idx)
    #         show_image(dr)

        


    pred = torch.rand((8, 30, 7, 7))
    b, _, _, _ = pred.shape

    class_prob_maps = get_class_probability_maps(pred)
    for idx in range(b):
        vis = visualize_class_probability_maps(class_prob_maps=class_prob_maps, image=image, idx=idx)
        show_image(vis)
        save_image(
            img=vis,
            path=f"""/Users/jongbeomkim/Desktop/workspace/segmentation_and_detection/yolo/class_probability_maps/{idx + 1}.jpg"""
        )

    for idx in range(b):
        dr = draw_predicted_bboxes(pred=pred, image=image, idx=idx)
        # show_image(dr)
        save_image(
                img=dr,
                path=f"""/Users/jongbeomkim/Desktop/workspace/segmentation_and_detection/yolo/sample_predicted_bboxes/{idx + 1}.jpg"""
            )
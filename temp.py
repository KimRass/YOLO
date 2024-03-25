import torch

# Define the size of your tensor
batch_size = 3
tensor_size = (batch_size, 5, 5)  # Example size, you can adjust according to your needs

# Define indices where you want to put 1 for each batch
indices = [
    [(1, 0), (3, 4)],  # Indices for the first batch
    [(0, 2), (4, 3)],  # Indices for the second batch
    [(2, 1), (3, 0)]   # Indices for the third batch
] 

# Initialize tensor with zeros
tensor = torch.zeros(tensor_size)

# Convert indices to tensor
indices_tensor = torch.tensor(indices)
indices_tensor.shape
indices_tensor[0]

# Create a mask tensor with shape (batch_size, 5, 5) and fill the specified indices with 1
mask = torch.zeros(tensor_size)
mask.scatter_(1, indices_tensor[:, :, 0].unsqueeze(2), 1)
mask.scatter_(2, indices_tensor[:, :, 1].unsqueeze(1), 1)

# Apply the mask to the tensor
tensor += mask

print(tensor)



def _encode(self, coord_gt, cls_gt):
        img_size=448
        # a = (((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2) % cell_size) / cell_size
        # b = (((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2) % cell_size) / cell_size
        # c = (coord_gt[:, :, 2] - coord_gt[:, :, 0]) / img_size
        # d = (coord_gt[:, :, 3] - coord_gt[:, :, 1]) / img_size
        # torch.cat([a, b, c, d], dim=2)

    def _encode(self, coord_gt, cls_gt):
        cell_size = 64
        img_size=448
        obj_mask = get_obj_mask(coord_gt, cls_gt)
        pred_coord[obj_mask].shape
        obj_mask
        coord_gt
        cls_gt
        obj_ms


        
        # "We parametrize the bounding box x and y coordinates to be offsets
        # of a particular grid cell location so they are also bounded between 0 and 1."
        gt["x"] = gt.apply(
            lambda x: (((x["l"] + x["r"]) / 2) % self.cell_size) / self.cell_size,
            axis=1
        )
        gt["y"] = gt.apply(
            lambda x: (((x["t"] + x["b"]) / 2) % self.cell_size) / self.cell_size,
            axis=1
        )
        # "We normalize the bounding box width and height by the image width and height
        # so that they fall between 0 and 1."
        gt["w"] = gt.apply(lambda x: (x["r"] - x["l"]) / self.img_size, axis=1)
        gt["h"] = gt.apply(lambda x: (x["b"] - x["t"]) / self.img_size, axis=1)

        gt["x_grid"] = gt.apply(
            lambda x: int((x["l"] + x["r"]) / 2 / self.cell_size), axis=1
        )
        gt["y_grid"] = gt.apply(
            lambda x: int((x["t"] + x["b"]) / 2 / self.cell_size), axis=1
        )
        return gt

    def decode(self, x):
        # x = out
        bbox = x.clone()

        bbox[:, (2, 7), ...] *= self.img_size # w
        bbox[:, (3, 8), ...] *= self.img_size # h
        bbox[:, (0, 5), ...] *= self.cell_size # x
        bbox[:, (0, 5), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(0) # x
        bbox[:, (1, 6), ...] *= self.cell_size # y
        bbox[:, (1, 6), ...] += torch.linspace(
            0, self.img_size - self.cell_size, self.n_cells,
        ).unsqueeze(1) # y

        l = bbox[:, (0, 5), ...] - bbox[:, (2, 7), ...] / 2
        t = bbox[:, (1, 6), ...] - bbox[:, (3, 8), ...] / 2
        r = bbox[:, (0, 5), ...] + bbox[:, (2, 7), ...] / 2
        b = bbox[:, (1, 6), ...] + bbox[:, (3, 8), ...] / 2

        bbox[:, (0, 5), ...] = l
        bbox[:, (1, 6), ...] = t
        bbox[:, (2, 7), ...] = r
        bbox[:, (3, 8), ...] = b

        bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...] = torch.clip(
            bbox[:, (0, 1, 2, 3, 5, 6, 7, 8), ...], min=0, max=self.img_size
        )

        bbox1 = torch.cat([bbox[:, : 5, ...], bbox[:, 10:, ...]], dim=1)
        bbox1 = rearrange(bbox1, pattern="b c h w -> b (h w) c")

        bbox2 = torch.cat([bbox[:, 5: 10, ...], bbox[:, 10:, ...]], dim=1)
        bbox2 = rearrange(bbox2, pattern="b c h w -> b (h w) c")

        bbox = torch.cat([bbox1, bbox2], dim=1)
        # return torch.cat([bbox[..., : 5], bbox[..., 5:]], dim=2)
        return bbox[..., : 4], bbox[..., 4], bbox[..., 5:]


def get_obj_mask(coord_gt, cls_gt):
        cell_size = 64
        batch_size = cls_gt.size(0)
        x_idx = ((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2 // cell_size).long()
        y_idx = ((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2 // cell_size).long()
        appears = torch.zeros(size=(batch_size, 7, 7), dtype=torch.bool)
        for batch_idx in range(batch_size):
            for bbox_idx in range(cls_gt.size(1)):
                cls_idx = cls_gt[batch_idx, bbox_idx].item()
                if cls_idx != 20:
                    appears[batch_idx, x_idx[batch_idx, bbox_idx], y_idx[batch_idx, bbox_idx]] = True
        return appears


def get_obj_mask(self, coord_gt, cls_gt):
        batch_size = cls_gt.size(0)
        x_idx = ((coord_gt[:, :, 0] + coord_gt[:, :, 2]) / 2 // self.cell_size).long()
        y_idx = ((coord_gt[:, :, 1] + coord_gt[:, :, 3]) / 2 // self.cell_size).long()
        
        appears = torch.zeros(size=(batch_size, (self.n_cells ** 2) * 2), dtype=torch.bool)
        idx = (x_idx * self.n_cells + y_idx)
        appears[
            torch.arange(batch_size).repeat_interleave(cls_gt.size(1))[(cls_gt != 20).view(-1)],
            idx.view(-1)[(cls_gt != 20).view(-1)],
        ] = 1
        appears[
            torch.arange(batch_size).repeat_interleave(cls_gt.size(1))[(cls_gt != 20).view(-1)],
            idx.view(-1)[(cls_gt != 20).view(-1)] + self.n_cells ** 2,
        ] = 1
        return appears


        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DynamicPadding(object):
    def __call__(self, batch):
        images = list()
        gt_ltrbs = list()
        gt_cls_idxs = list()
        max_n_objs = 0
        for image, gt_ltrb, gt_cls_idx in batch:
            images.append(image)
            gt_ltrbs.append(gt_ltrb)
            gt_cls_idxs.append(gt_cls_idx)

            n_objs = gt_ltrb.size(0)
            if n_objs > max_n_objs:
                max_n_objs = n_objs

        image = torch.stack(images)
        gt_ltrb = torch.stack(
            [
                torch.cat(
                    [
                        gt_ltrb,
                        # torch.full(
                        #     size=(max_n_objs - gt_ltrb.size(0), 4),
                        #     fill_value=len(VOC_CLASSES),
                        #     dtype=torch.int32,
                        # )
                        torch.zeros(
                            size=(max_n_objs - gt_ltrb.size(0), 4), dtype=torch.int32,
                        ),
                    ],
                    dim=0,
                )
                for gt_ltrb
                in gt_ltrbs
            ]
        )
        gt_cls_idx = torch.stack(
            [
                torch.cat(
                    [
                        gt_cls_idx,
                        torch.full(
                            size=(max_n_objs - gt_cls_idx.size(0),),
                            fill_value=len(VOC_CLASSES),
                            dtype=torch.int32,
                        )
                    ],
                    dim=0,
                )
                for gt_cls_idx
                in gt_cls_idxs
            ]
        )
        return image, gt_ltrb, gt_cls_idx


def obj_idx_to_obj_mask(obj_idx):
    mask = torch.zeros(size=(7, 7), dtype=torch.bool)
    mask[obj_idx[:, 0], obj_idx[:, 1]] = True
    return mask


def obj_idx_to_noobj_mask(obj_idx):
    obj_mask = obj_idx_to_obj_mask(obj_idx)
    return ~obj_mask[None, ...].repeat(2, 1, 1)

def get_dedup_row_idx(row_idx, n_bboxes_per_cell=1):
    cnts = defaultdict(int)
    valid_indices = list()
    for idx in range(row_idx.size(0)):
        row = row_idx[idx].item()
        if cnts[row] < n_bboxes_per_cell:
            valid_indices.append(idx)
        cnts[row] += 1
    return valid_indices

def row_idx_to_noobj_mask(self, row_idx):
    """
    "$\mathbb{1}^{noobj}_{ij}$";
    """
    obj_mask = self.row_idx_to_obj_mask(row_idx)
    return ~obj_mask


    # cell_size = 64
    # n_cells= 7
    # n_bboxes_per_cell = 1
    # for i in range(10000):
    #     image, gt_ltrb, gt_cls_idx = ds[i]
    #     gt_xywh = ltrb_to_xywh(gt_ltrb)
    #     obj_idx = xy_to_obj_idx(gt_xywh[:, : 2])
    #     if obj_idx.size(0) != torch.unique(obj_idx, dim=0).size(0):
    #         break
    
    
    # xml_path = "/home/dmeta0304/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations/2012_004272.xml"
        # cell_size = 64
        # n_bboxes_per_cell=2

        # for bbox in xroot.findall("object"):
        #     l = int(bbox.find("bndbox").find("xmin").text)
        #     t = int(bbox.find("bndbox").find("ymin").text)
        #     r = int(bbox.find("bndbox").find("xmax").text)
        #     b = int(bbox.find("bndbox").find("ymax").text)

        #     x = (l + r) // 2
        #     y = (t + b) // 2
        #     w = r - l
        #     h = b - t

        #     x_cell_idx = x // cell_size
        #     y_cell_idx = y // cell_size
        #     x_cell_idx, y_cell_idx
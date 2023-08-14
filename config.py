import torch

### Architecture
LEAKY_RELU_SLOPE = 0.1

### Data
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
IMG_SIZE = 448
# "We introduce random scaling and translations of up to 20% of the original image size."
TRANSFORM_RATIO = 0.2
N_BBOXES = 2
N_CELLS = 7
CELL_SIZE = IMG_SIZE // N_CELLS
ANNOT_DIR = "/home/user/cv/voc2012/VOCdevkit/VOC2012/Annotations"

## Loss function
LAMB_COORD = 5
LAMB_NOOBJ = 0.5

### Optimizer
INIT_LR = 1e-3
# "A momentum of 0.9 and a decay of 0.0005."
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

### Training
SEED = 17
N_WORKERS = 4
MULTI_GPU = True
AUTOCAST = True
N_GPUS = torch.cuda.device_count()
BATCH_SIZE = 64 # "We use a batch size of 64."
N_EPOCHS = 135 # "We train the network for about 135 epochs."
N_PRINT_EPOCHS = 1
N_CKPT_EPOCHS = 10

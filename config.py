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
ANNOT_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/Annotations"

## Loss function
LAMB_COORD = 5
LAMB_NOOBJ = 0.5

### Optimizer
# "A momentum of 0.9 and a decay of 0.0005."
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

### Training
SEED = 17
N_WORKERS = 4
MULTI_GPU = True # "We use a batch size of 64."
BATCH_SIZE = 64
N_EPOCHS = 135 # "We train the network for about 135 epochs."
AUTOCAST = True
N_PRINT_STEPS = 1000
N_CKPT_STEPS = 4000

import torch # 파이토치
import torch.nn as nn # 레이어, 활성화함수 등 basic building blocks이 들어있습니다.
import torchvision.transforms as transforms # 변환 연산(resize 등)이 들어있는 모듈입니다. 
from torchvision.datasets import VOCDetection # PASCAL VOC 2007을 가져오는데 쓰입니다. 클래스 형식으로 만들어져 사용을 편리하게 할 
import xmltodict # xml파일의 내용을 딕셔너리에 저장할 수 있는 메소드들이 들어있는 모듈입니다. 
from PIL import Image # 이미지를 읽기 위해 사용합니다
import numpy as np # 넘파이, 저는 넘파이로 연산하는게 익숙해 넘파이를 사용합니다.
from tqdm import tqdm # tqdm, for문의 진행상황을 보기 위해 사용합니다.


class YOLO_PASCAL_VOC(VOCDetection):
    def __getitem__(self, index):
        img = (Image.open(self.images[index]).convert('RGB')).resize((224, 224))
        img_transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))])
        img = torch.divide(img_transform(img), 255)


        target = xmltodict.parse(open(self.annotations[index]).read())

        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                   "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]

        label = np.zeros((7, 7, 25), dtype = float)

        Image_Height = float(target['annotation']['size']['height'])
        Image_Width  = float(target['annotation']['size']['width'])

        # 바운딩 박스 정보 받아오기
        try:
            for obj in target['annotation']['object']:
                
                # class의 index 휙득
                class_index = classes.index(obj['name'].lower())
                
                # min, max좌표 얻기
                x_min = float(obj['bndbox']['xmin']) 
                y_min = float(obj['bndbox']['ymin'])
                x_max = float(obj['bndbox']['xmax']) 
                y_max = float(obj['bndbox']['ymax'])

                # 224*224에 맞게 변형시켜줌
                x_min = float((224.0/Image_Width)*x_min)
                y_min = float((224.0/Image_Height)*y_min)
                x_max = float((224.0/Image_Width)*x_max)
                y_max = float((224.0/Image_Height)*y_max)

                # 변형시킨걸 x,y,w,h로 만들기 
                x = (x_min + x_max)/2.0
                y = (y_min + y_max)/2.0
                w = x_max - x_min
                h = y_max - y_min

                # x,y가 속한 cell알아내기
                x_cell = int(x/32) # 0~6
                y_cell = int(y/32) # 0~6
                # cell의 중심 좌표는 (0.5, 0.5)다
                x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
                y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

                # w, h 를 0~1 사이의 값으로 만들기
                w = w / 224.0
                h = h / 224.0

                class_index_inCell = class_index + 5

                label[y_cell][x_cell][0] = x_val_inCell
                label[y_cell][x_cell][1] = y_val_inCell
                label[y_cell][x_cell][2] = w
                label[y_cell][x_cell][3] = h
                label[y_cell][x_cell][4] = 1.0
                label[y_cell][x_cell][class_index_inCell] = 1.0


        # single-object in image
        except TypeError as e : 
            # class의 index 휙득
            class_index = classes.index(target['annotation']['object']['name'].lower())
                
            # min, max좌표 얻기
            x_min = float(target['annotation']['object']['bndbox']['xmin']) 
            y_min = float(target['annotation']['object']['bndbox']['ymin'])
            x_max = float(target['annotation']['object']['bndbox']['xmax']) 
            y_max = float(target['annotation']['object']['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((224.0/Image_Width)*x_min)
            y_min = float((224.0/Image_Height)*y_min)
            x_max = float((224.0/Image_Width)*x_max)
            y_max = float((224.0/Image_Height)*y_max)

            # 변형시킨걸 x,y,w,h로 만들기 
            x = (x_min + x_max)/2.0
            y = (y_min + y_max)/2.0
            w = x_max - x_min
            h = y_max - y_min

            # x,y가 속한 cell알아내기
            x_cell = int(x/32) # 0~6
            y_cell = int(y/32) # 0~6
            x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
            y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

            # w, h 를 0~1 사이의 값으로 만들기
            w = w / 224.0
            h = h / 224.0

            class_index_inCell = class_index + 5

            label[y_cell][x_cell][0] = x_val_inCell
            label[y_cell][x_cell][1] = y_val_inCell
            label[y_cell][x_cell][2] = w
            label[y_cell][x_cell][3] = h
            label[y_cell][x_cell][4] = 1.0
            label[y_cell][x_cell][class_index_inCell] = 1.0
            
        return img, torch.tensor(label)

ds = YOLO_PASCAL_VOC(root="/Users/jongbeomkim/Documents/datasets/voc2012", year="2012", image_set="train")
type(ds[0])
len(ds[0])
a, b = ds[0]
a.shape, b.shape
len(b)
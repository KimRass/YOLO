# Object Detection
- References:
    - https://machinethink.net/blog/object-detection/
- Each box also has a confidence score that says how likely the model thinks this box really contains an object.
- The score of 52.14% given here is a combination of the class score, which was 82.16% dog, and a confidence score of how likely it is that the bounding box really contains an object, which was 63.47%.

# Paper Reading
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
## Methodology
- Formally we define confidence as
$$P(Object) \cdot IOU^{gt}_{pred}$$
- (Comment: ground truth bounding box와의 IoU가 낮더라도 오브젝트가 존재할 확률이 높거나 오브젝트가 존재할 확률이 높더라도 ground truth bounding box와의 IoU가 높으면 confidence는 높은 값을 가집니다.)
- ***If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.***
- ***Each bounding box consists of 5 predictions: x, y, w, h and confidence.***
    - The $(x, y)$ coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. ***We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.***
- Finally the confidence prediction represents the IOU between the predicted box and any ground truth box. Each grid cell also predicts $C$ conditional class probabilities, $P(Class \mid Object)$. These probabilities are conditioned on the grid cell containing an object. (Comment: 이 수식의 의미는, '해당 그리드 셀에 오브젝트가 존재한다면' 그 오브젝트의 클래스가 $Class$일 확률입니다.) ***We only predict one set of class probabilities per grid cell, regardless of the number of boxes*** $B$***.*** It divides the image into an $S \times S$ grid and ***for each grid cell predicts*** $B$ ***bounding boxes, confidence for those boxes, and*** $C$ ***class probabilities. These predictions are encoded as an*** $S \times S \times (B \times 5 + C)$ ***tensor.***
## Loss
$$
\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} \bigg[(x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} + (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} + (C_{i} - \hat{C}_{i})^{2}\bigg]
\\+ \sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes} \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}
\\+ \lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} 1^{noobj}_{ij} \big(C_{i} - \hat{C}_{i}\big)^{2}
$$
- $\mathbb{1}^{obj}_{i}$: ***Denotes if object appears in cell*** $i$
- $\mathbb{1}^{obj}_{ij}$: ***Denotes that the*** $j$ th ***bounding box predictor in cell*** $i$ ***is "responsible" for that prediction.***
- (Comment: $\mathbb{1}^{noobj}_{ij}$: $\mathbb{1}^{obj}_{ij}$가 0일 때 1을 가지며 그 반대의 경우도 마찬가지입니다.)
- (위 loss의 수식에서 $j$가 빠져있는데, $j$는 각 그리드 셀마다 $B$개의 predictors 중에서 confidence가 가장 높은 것으로 선택됩니다.)
- We use sum-squared error because it is easy to optimize, however ***it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the "confidence" scores of those cells towards zero, often overpowering the gradient from cells that do contain objects.*** This can lead to model instability, causing training to diverge early on. To remedy this, ***we increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects. We use two parameters,*** $\lambda_{coord}$ ***and*** $\lambda_{noobj}$ ***to accomplish this. We set*** $\lambda_{coord} = 5$ ***and*** $\lambda_{noobj} = 0.5$***.***
- Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly. At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be "responsible" for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall. During training we optimize the following, multi-part
- ***Note that the loss function only penalizes classification error if an object is present in that grid cell. It also only penalizes bounding box coordinate error if that predictor is "irresponsible" for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).***

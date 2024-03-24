# 1. Theorectical Background
- Comment: 이 수식의 의미는, '해당 그리드 셀에 오브젝트가 존재한다면' 그 오브젝트의 클래스가 $Class$일 확률입니다.
- Comment: ground truth bounding box와의 IoU가 낮더라도 오브젝트가 존재할 확률이 높거나 오브젝트가 존재할 확률이 높더라도 ground truth bounding box와의 IoU가 높으면 confidence는 높은 값을 가집니다.
$$P(Object) \cdot IOU^{gt}_{pred}$$
$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} \bigg[(x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} + (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} + (C_{i} - \hat{C}_{i})^{2}\bigg]
\\+ \sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes} \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}
\\+ \lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} 1^{noobj}_{ij} \big(C_{i} - \hat{C}_{i}\big)^{2}$$
- $\mathbb{1}^{obj}_{i}$: ***Denotes if object appears in cell*** $i$
- $\mathbb{1}^{obj}_{ij}$: ***Denotes that the*** $j$ th ***bounding box predictor in cell*** $i$ ***is "responsible" for that prediction.***
<!-- - (Comment: $\mathbb{1}^{noobj}_{ij}$: $\mathbb{1}^{obj}_{ij}$가 0일 때 1을 가지며 그 반대의 경우도 마찬가지입니다.) -->
<!-- - (위 loss의 수식에서 $j$가 빠져있는데, $j$는 각 그리드 셀마다 $B$개의 predictors 중에서 confidence가 가장 높은 것으로 선택됩니다.) -->

$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} \bigg[ (x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} \bigg]$$
$$\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} \bigg[ (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} \bigg]$$
$$\sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} (C_{i} - \hat{C}_{i})^{2}$$
$$\lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} 1^{noobj}_{ij} \big( C_{i} - \hat{C}_{i} \big)^{2}$$
$$\sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes} \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}$$

# 2. References
- [1] https://machinethink.net/blog/object-detection/

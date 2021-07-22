# EfficientNetV2 in TensorFlow
This repo is a reimplementation of EfficientNet V2. The code base is heavily inspired by [TensorFlow implementation](https://github.com/google/automl/tree/master/efficientnetv2) and [EfficientNet Keras](https://github.com/qubvel/efficientnet) 
<img src="https://raw.githubusercontent.com/google/automl/master/efficientnetv2/g3doc/param_flops.png">
## Examples
___
* Start the model and reload weights
```python
from efficientnetv2 import EfficientNetV2_L
model = EfficientNetV2_L(input_shape=(512, 512, 3), weights=None)
model.load_weights('path_to_model.h5')
```
## Architectures
|Model|params|ImageNet Acc|CIFAR-10|
|---|:---:|:---:|:---:|
|EfficientNetV2_Base|7M|-|-|
|EfficientNetV2_S|22M|83.9|98.7|
|EfficientNetV2_M|54M|85.1|99.0|
|EfficientNetV2_L|120M|85.7|99.1|
|EfficientNetV2_XL (21K)|208M|87.3|-|

## Installation
___
### Requirements
* TensorFlow >= 2.4
### From source
```bash
pip install -U git+https://github.com/GdoongMathew/EfficientNetV2
```

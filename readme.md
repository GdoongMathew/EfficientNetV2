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
## Pretrained EfficientNetV2 Weights
Ported from automl efficientnetv2 imagenet21k pretrained weights

|Arch|imagenet|imagenet21k|imagenet21k-ft1k|
|---|:---:|:---:|:---:|
|EfficientNetV2_S|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-s_imagenet.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-s_imagenet21k.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-s_imagenet21k-ft1k.h5)|
|EfficientNetV2_M|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-m_imagenet.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-m_imagenet21k.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-m_imagenet21k-ft1k.h5)|
|EfficientNetV2_L|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-l_imagenet.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-l_imagenet21k.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-l_imagenet21k-ft1k.h5)|
|EfficientNetV2_XL|-|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-xl_imagenet21k.h5)|[h5](https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/efficientnetv2-xl_imagenet21k-ft1k.h5)|
## Installation
___
### Requirements
* TensorFlow >= 2.4
### From source
```bash
pip install -U git+https://github.com/GdoongMathew/EfficientNetV2
```

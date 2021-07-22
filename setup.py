import os
import sys
import shutil

from setuptools import find_packages, setup, Command

NAME = "efficientnetv2"
DESCRIPTION = "EfficientNetV2 model reimplementation in TensorFlow Keras."
URL = "https://github.com/GdoongMathew/EfficientNetV2"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"

try:
    with open('requirements.txt', encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')

except:
    REQUIRED = []

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
)

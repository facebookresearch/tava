# Copyright (c) Meta Platforms, Inc. and affiliates.

import setuptools

INSTALL_REQUIREMENTS = [
    "imageio[freeimage]",
    "numpy==1.22.3",
    "opencv-python==4.5.5.64",
    "matplotlib==3.5.1",
    "scikit-image==0.19.2",
    "hydra-core==1.1.1",
    "tqdm==4.64.0",
    "matplotlib==3.5.1",
    "tensorboard==2.8.0",
    "plotly",
    "nbformat>=4.2.0",
    "protobuf",
    "chumpy",
    "scikit-learn-extra",
]

setuptools.setup(
    name="tava",
    url="https://github.com/facebookresearch/tava",
    description="TAVA: Template-free Animatable Volumetric Actors.",
    version="1.0.0",
    author="Ruilong Li",
    author_email="ruilongli94@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
)

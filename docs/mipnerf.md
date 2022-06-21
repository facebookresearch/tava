# Mip-NeRF

This document shows you how to run the original [Mip-NeRF](https://jonbarron.info/mipnerf/) using this codebase.

First you'll need to download the dataset
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip`.

Then you can just kick of the training with this command (Remember to change the `root_fp`):
```
CUDA_VISIBLE_DEVICES=8,9 python launch.py --config-name=mipnerf \
    dataset.root_fp=/home/ruilongli/data/nerf_synthetic/ \
    dataset.subject_id=lego
```
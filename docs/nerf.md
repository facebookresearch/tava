# NeRF

This document shows you how to run the original [NeRF](https://www.matthewtancik.com/nerf) using this codebase.

First you'll need to download the dataset
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip`.

Then you can just kick of the training with this command (Remember to change the `root_fp`):
```
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=nerf \
    dataset.root_fp=/home/ruilongli/data/nerf_synthetic/ \
    dataset.subject_id=lego
```
# Experiments

Before start, please download and process the data follow the instructions [docs/dataset.md](dataset.md). The directory structure of the data is expected to be:
```
data/
├── animal/
|   ├── Hare_male_full_RM/
|   └── Wolf_cub_full_RM_2/
└── zju/
    ├── SMPL_NEUTRAL.pkl
    ├── CoreView_313/
    ├── ...
    └── CoreView_386/
```


## A Quick Test

Before kicking off the training, I recommand you to do a dry-run of the code that can quickly test
if everything (data & dependency & multigpu) etc are set up correctly. You certainly don't want to see
a failure job after waiting for a few hours. A successful dry-run garentees your program can keep running
till the very end.
```
FASTARGS="max_steps=1 print_every=1 save_every=1 eval_every=1 dataset.resize_factor=0.1 hydra.run.dir=outputs/dryrun"
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn dataset=zju pos_enc=snarf $FASTARGS
```


## Our Method: TAVA

    "TAVA: Template-free Animatable Volumetric Actors."

The default output directory is:
`./outputs/dynamic/<dataset>/<subject_id>/snarf/`. You can check the on-the-fly qualitative evalution results in the folder `eval_imgs_otf` and quantitative scores in the `val_xxx_metrics_otf.txt` file. There is also a tensorboard log file in this folder for you to check on the loss curves.

### Training.

```
# train the ZJU subjects. (full model)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=313 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1

# train the animal subjects. (LBS residual and AO shading are both disabled)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal_hare \
    model.shading_mode=null \
    pos_enc=snarf \
    pos_enc.offset_net_enabled=false \
    loss_bone_w_mult=1.0

## train the ZJU subjects. (full model but using NeRF instead of MipNeRF)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=nerf_dyn \
    dataset=zju \
    dataset.subject_id=313 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 \
    model.num_levels=1 model.num_samples_coarse=128
```

## Baselines: NARF

    "Neural Articulated Radiance Field, ICCV 2021."

```
# train the ZJU subjects.
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=313 \
    pos_enc=narf

# train the animal subjects.
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal_hare \
    model.shading_mode=null \
    pos_enc=narf
```

## Baselines: More are coming out.
# Experiments

We experimented on 4 subjects in the ZJUMocap data and 2 animal subject Hare and Wolf. Please download the processed data from [GoogleDrive](https://drive.google.com/drive/folders/14yYd03FFmg3F64nMj8jUkioLASoPhwKg?usp=sharing), and put it under `$DATA` folder.


## A Quick Test

Before kicking off the training, I recommand you to do a dry-run of the code that can quickly test
if everything (data & dependency & multigpu) etc are set up correctly. You certainly don't want to see
a failure job after waiting for a few hours. A successful dry-run can garentee your program can keep running
till the very end.
```
FASTARGS="max_steps=1 print_every=1 save_every=1 eval_every=1 dataset.resize_factor=0.1 hydra.run.dir=outputs/dryrun"
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.root_fp=$DATA \
    pos_enc=snarf \
    $FASTARGS
```


## Our Method: TAVA

    "TAVA: Template-free Animatable Volumetric Actors."

The default output directory is:
`./outputs/dynamic/<dataset>/<subject_id>/snarf/`. You can check the on-the-fly quantitative evalution results in the folder `eval_imgs_otf` and qualitative scores in the `val_xxx_metrics_otf.txt` file. There is also a tensorboard log file in this folder for you to check on the loss curves.

### Training.

```
# train the ZJU subjects. (full model)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.root_fp=$DATA \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1

# train the animal subjects. (LBS residual and shading are both disabled)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal \
    dataset.root_fp=$DATA \
    model.shading_mode=null \
    pos_enc=snarf \
    pos_enc.offset_net_enabled=false \
    loss_bone_w_mult=1.0

## train the ZJU subjects. (full model but using NeRF instead of MipNeRF)
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=nerf_dyn \
    dataset=zju \
    dataset.root_fp=$DATA \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 \
    model.num_levels=2 model.num_samples_coarse=32 model.num_samples_fine=64
```

### Evaluation.

Keep the same arguments when you do the training which will be used to locate the directory of the checkpoints, and add some additional arguments to it: `engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=50`.

```
# evaluate on animal
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal \
    dataset.root_fp=$DATA \
    model.shading_mode=null \
    pos_enc=snarf \
    pos_enc.offset_net_enabled=false \
    loss_bone_w_mult=1.0 \
    engine=evaluator \
    eval_cache_dir=eval_imgs \
    compute_metrics=true \
    resume=true
```

## Baselines: NARF

    "Neural Articulated Radiance Field, ICCV 2021."

```
# train the ZJU subjects.
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.root_fp=$DATA \
    pos_enc=narf

# train the animal subjects.
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal \
    dataset.root_fp=$DATA \
    model.shading_mode=null \
    pos_enc=narf
```

For evaluation, the instructions are same as above. Simply add some additional arguments `engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=50` to run on the validation sets.

## Baselines: More are coming out.
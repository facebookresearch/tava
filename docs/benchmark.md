# Benchmarks

This page shows instructions on how to replicate the results in our paper.

Please first check the instructions [doc/experiment.md](experiment.md) on setting up the dataset and conducting a dryrun test of the code base to make sure everything is set up correctly.

## Training.

Simplly following [doc/experiment.md](experiment.md) on training our model as well as baselines.


## Evaluation.

Please note our paper evaluate 100 images on each set (`eval_per_gpu=50` with two gpus). If you are not using two gpus, you would need to change the `eval_per_gpu` accordingly to match with our paper (for example `eval_per_gpu=25` for four gpus)

### Using self-trained model.

Keep the same arguments when you do the training which will be used to automatically locate the directory of the checkpoints, and add some additional arguments to it: `engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=50`.

For example, for animal Hare, we train TAVA method with command line:
```
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal_hare \
    model.shading_mode=null \
    pos_enc=snarf \
    pos_enc.offset_net_enabled=false \
    loss_bone_w_mult=1.0
```

Then we evaluate it with:
```
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal_hare \
    model.shading_mode=null \
    pos_enc=snarf \
    pos_enc.offset_net_enabled=false \
    loss_bone_w_mult=1.0 \
    engine=evaluator \
    eval_cache_dir=eval_imgs \
    compute_metrics=true \
    resume=true \
    eval_per_gpu=50
```

This rule applies to also baselines methods such as NARF.


### Using released model.


**Download model:** For animal subjects Hare and Wolf, we provide released model on our [project website](https://www.liruilong.cn/projects/tava/) for both our method as well as baselines (currently NARF supported). For legal issues, we won't be able to provide released model for ZJU mocap dataset. Please refer to [doc/experiment.md](experiment.md) on how to train the model from scratch.

To evaluate, please keep the same arguments when you do the training, and add some additional arguments to it: `engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=50` for evaluation. Then further set where you put the released model with `hydra.run.dir=<directory-of-the-checkpoints-folder>`. For example, to evaluate baseline method NARF on animal Hare:

```
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    dataset=animal_hare \
    model.shading_mode=null \
    pos_enc=narf \
    engine=evaluator \
    eval_cache_dir=eval_imgs \
    compute_metrics=true \
    resume=true \
    eval_per_gpu=50 \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_hare/Hare_male_full_RM/narf/ 
```

**Please see [tools/eval_psnr.sh](../tools/eval_psnr.sh) for the complete commands we used for all the experiments. It will produce the exact numbers shown in the paper:**
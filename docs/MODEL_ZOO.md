# Model Zoo and Baselines

This page shows instructions on how to replicate the results in our paper.

Please first check the instructions [here](EXPERIMENT.md) on setting up the dataset and conducting a dryrun test of the code base to make sure everything is set up correctly.

## Download the Checkpoints.

You can either train our model from scratch following the instructions [here](EXPERIMENT.md) to get the checkpoints, or download our checkpoints from here:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Subject</th>
<th valign="bottom">Method</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ZJU-Mocap</td>
<td align="center">313</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/15lAdlcYTlKoto2krOBcQvcDF3cBfgisD/view?usp=sharing">model</a></td>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">315</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/1gnhryGJETm6UcbQn-UADtjFCwYGo7Yeg/view?usp=sharing">model</a></td>
</tr>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">377</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/1tf-Q-lNCPAJ237dX_uOP8hdzivxapcMc/view?usp=sharing">model</a></td>
</tr>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">386</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/13bxlHsmXE9pdbjCAqnYj1hWQKt70yH2k/view?usp=sharing">model</a></td>
</tr>

<tr>
<td align="left">ZJU-Mocap</td>
<td align="center">313</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/19mfhH1eNp17MCX1XpcrSFSpMECg5f4ql/view?usp=sharing">model</a></td>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">315</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/18YN1Pb54wdNkL5y8GPeIC5n69ReSfKtz/view?usp=sharing">model</a></td>
</tr>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">377</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/15kzxp0EA3hlAD6_P1HEYxMNWf9cW3K4f/view?usp=sharing">model</a></td>
</tr>
</tr>
<td align="left">ZJU-Mocap</td>
<td align="center">386</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/1nOfKaWkQ2uBDbjoL8UknYoBHMNDrZWUE/view?usp=sharing">model</a></td>
</tr>

<tr>
<td align="left">Forest & Friends (Animal)</td>
<td align="center">hare</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/1C-Hn38-byYn7AA2iBwIyce0gCy1jlzOS/view?usp=sharing">model</a></td>
</tr>
<tr>
<td align="left">Forest & Friends (Animal)</td>
<td align="center">wolf</td>
<td align="center">TAVA</td>
<td align="center"><a href="https://drive.google.com/file/d/1xY90xUrDRZxPmcRpiNF5pV1YyLVNMvRR/view?usp=sharing">model</a></td>
</tr>

<tr>
<td align="left">Forest & Friends (Animal)</td>
<td align="center">hare</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/1C-Hn38-byYn7AA2iBwIyce0gCy1jlzOS/view?usp=sharing">model</a></td>
</tr>
<tr>
<td align="left">Forest & Friends (Animal)</td>
<td align="center">wolf</td>
<td align="center">NARF</td>
<td align="center"><a href="https://drive.google.com/file/d/1xY90xUrDRZxPmcRpiNF5pV1YyLVNMvRR/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

All the above checkpoints are in Google Drive at [here](https://drive.google.com/drive/folders/1GyfVXUmhPrYu6tZkoTGN5q9OttW489cc?usp=sharing), in case you want to download them all at once.

## Evaluation and Replication.

First you would need to setup some arguments for the command line. Please note our paper evaluate 100 images on
each set (`eval_per_gpu=50` with two gpus). If you are not using two gpus, you would need to change the `eval_per_gpu`
accordingly to match with our paper.

```
# dataset
ARGS_ANIMAL_WOLF="dataset=animal_wolf dataset.root_fp=/home/ruilongli/data/forest_and_friends_rendering/"
ARGS_ANIMAL_HARE="dataset=animal_hare dataset.root_fp=/home/ruilongli/data/forest_and_friends_rendering/"
ARGS_ZJU_313="dataset=zju dataset.subject_id=313 dataset.root_fp=/home/ruilongli/data/zju_mocap/neuralbody/"
ARGS_ZJU_315="dataset=zju dataset.subject_id=315 dataset.root_fp=/home/ruilongli/data/zju_mocap/neuralbody/"
ARGS_ZJU_377="dataset=zju dataset.subject_id=377 dataset.root_fp=/home/ruilongli/data/zju_mocap/neuralbody/"
ARGS_ZJU_386="dataset=zju dataset.subject_id=386 dataset.root_fp=/home/ruilongli/data/zju_mocap/neuralbody/"

# methods
ARGS_TAVA_ANIMAL="pos_enc=snarf loss_bone_w_mult=1.0 pos_enc.offset_net_enabled=false model.shading_mode=null"
ARGS_TAVA_ZJU="pos_enc=snarf loss_bone_w_mult=1.0 pos_enc.offset_net_enabled=true model.shading_mode=implicit_AO"
ARGS_NARF="pos_enc=narf model.shading_mode=null"

# evaluation (for two gpus)
ARGS_EVAL="engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=50"
```

For NARF evaluation:

```
# NARF + animal hare
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ANIMAL_HARE $ARGS_NARF \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_hare/Hare_male_full_RM/narf/ 

# NARF + animal wolf
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ANIMAL_WOLF $ARGS_NARF \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_wolf/Wolf_cub_full_RM_2/narf/

# NARF + ZJU 313
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ZJU_313 $ARGS_NARF \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/313/narf/

# NARF + ZJU 315
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ZJU_315 $ARGS_NARF \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/315/narf/

# Similar for ZJU 377 386
```


For TAVA evaluation:

```
# TAVA + animal hare
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ANIMAL_HARE $ARGS_TAVA_ANIMAL \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_hare/Hare_male_full_RM/snarf/

# TAVA + animal wolf
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ANIMAL_WOLF $ARGS_TAVA_ANIMAL \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_wolf/Wolf_cub_full_RM_2/snarf/

# TAVA + ZJU 313
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ZJU_313 $ARGS_TAVA_ZJU \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/313/snarf/

# TAVA + ZJU 315
CUDA_VISIBLE_DEVICES=0,1 python launch.py --config-name=mipnerf_dyn \
    $ARGS_EVAL $ARGS_ZJU_315 $ARGS_TAVA_ZJU \
    hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/315/snarf/

# Similar for ZJU 377 386
```

Other baselines are comming out.

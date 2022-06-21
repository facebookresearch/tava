# This script logs how to do psnr eval with released model,
# indicated with `hydra.run.dir=...`. For self-trained model,
# you can simplly remove the argument of `hydra.run.dir=...`,
# in which case the script will locate the checkpoints automatically.

# args for dataset
ARGS_ANIMAL_WOLF="dataset=animal_wolf"
ARGS_ANIMAL_HARE="dataset=animal_hare"
ARGS_ZJU_313="dataset=zju dataset.subject_id=313"
ARGS_ZJU_315="dataset=zju dataset.subject_id=315"
ARGS_ZJU_377="dataset=zju dataset.subject_id=377"
ARGS_ZJU_386="dataset=zju dataset.subject_id=386"

# args for methods
ARGS_TAVA_ANIMAL="pos_enc=snarf loss_bone_w_mult=1.0 pos_enc.offset_net_enabled=false model.shading_mode=null"
ARGS_TAVA_ZJU="pos_enc=snarf loss_bone_w_mult=1.0 pos_enc.offset_net_enabled=true model.shading_mode=implicit_AO"
ARGS_NARF="pos_enc=narf model.shading_mode=null"

# args for evaluation (for 4 gpus)
ARGS_EVAL="engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=25"

###########
# NARF
###########

# # NARF + animal hare (checking)
# CUDA_VISIBLE_DEVICES=5,6,7,8 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ANIMAL_HARE $ARGS_NARF \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_hare/Hare_male_full_RM/narf/ 

# # NARF + animal wolf
# CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ANIMAL_WOLF $ARGS_NARF \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_wolf/Wolf_cub_full_RM_2/narf/

# # NARF + ZJU 313
# CUDA_VISIBLE_DEVICES=5,6,7,8 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ZJU_313 $ARGS_NARF \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/313/narf/

# # NARF + ZJU 315
# CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ZJU_315 $ARGS_NARF \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/315/narf/

# # Similar for ZJU 377 386


###########
# TAVA
###########

# # TAVA + animal hare
# CUDA_VISIBLE_DEVICES=5,6,7,8 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ANIMAL_HARE $ARGS_TAVA_ANIMAL \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_hare/Hare_male_full_RM/snarf/

# # TAVA + animal wolf
# CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ANIMAL_WOLF $ARGS_TAVA_ANIMAL \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/animal_wolf/Wolf_cub_full_RM_2/snarf/

# TAVA + ZJU 313
# CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ZJU_313 $ARGS_TAVA_ZJU \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/313/snarf/

# # TAVA + ZJU 315
# CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py --config-name=mipnerf_dyn \
#     $ARGS_EVAL $ARGS_ZJU_315 $ARGS_TAVA_ZJU \
#     hydra.run.dir=/home/ruilongli/workspace/TAVA/outputs/release/zju_mocap/315/snarf/

# # Similar for ZJU 377 386
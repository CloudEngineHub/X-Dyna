# Copyright 2024 ByteDance and/or its affiliates.
#
# Copyright (2024) X-Dyna Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

export TORCH_USE_CUDA_DSA=1
if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-0}
job_name=${2-xdyna_infer}
pretrain_weight=${3-"SD/stable-diffusion-v1-5"} # path to pretrained SD1.5
output_dir=${4-"./output"} # save path
infer_config=${5-"configs/x_dyna.yaml"} # inference model config
pretrained_image_encoder_path=${6-"IP-Adapter/IP-Adapter/models/image_encoder"} # path to pretrained IP-Adapter image clip
pretrained_ipadapter_path=${7-"IP-Adapter/IP-Adapter/models/ip-adapter-plus_sd15.bin"} # path to pretrained IP-Adapter
pretrained_root_path=${8-"./pretrained_weights/initialization/unet_initialization"} # path to SD and IP-Adapter initialization root
test_data_file=${9-"examples/example.json"} # path to testing data file, used for batch inference
pose_controlnet_initialization_path=${10-"./pretrained_weights/initialization/controlnets_initialization/controlnet/control_v11p_sd15_openpose"}
pretrained_unet_path=${11-"./pretrained_weights/unet/"} # path to pretrained dynamics-adapter, motion module and unet
pretrained_controlnet_path=${12-"./pretrained_weights/controlnet/"} # path to pretrained pose controlnet
################

echo 'start job:' ${job_name}

now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}


# cfg 
export CUDA_VISIBLE_DEVICES=${gpus}
python inference_xdyna.py \
    --output ${output_dir} \
    --pretrain_weight "${pretrained_root_path}/${pretrain_weight}" \
    --length 192 \
    --height 896 --width 512 \
    --cfg 7.5 --infer_config ${infer_config} \
    --pretrained_image_encoder_path "${pretrained_root_path}/${pretrained_image_encoder_path}" \
    --pretrained_ipadapter_path "${pretrained_root_path}/${pretrained_ipadapter_path}" \
    --neg_prompt "" \
    --test_data_file ${test_data_file} \
    --pose_controlnet_initialization_path ${pose_controlnet_initialization_path} \
    --pretrained_unet_path ${pretrained_unet_path} \
    --pretrained_controlnet_path ${pretrained_controlnet_path} \
    --cross_id \
    --use_controlnet \
    --no_head_skeleton \
    --global_seed 40 \
    --stride 2 \
    --save_fps 15 \
    2>&1 | tee ${LOG_FILE}

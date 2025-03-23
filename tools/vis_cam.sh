#!/usr/bin/env bash
# export PYTHONPATH=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/mmaction:$PYTHONPATH
# set -x
# CONFIG=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/tsm_imagenet-pretrained-mobilenetv2_attnlast_8xb16-1x1x8-50e_kinetics400-rgb-dist/uni2tsm-attnlast-100epoch/epoch100_test/vis_data/config.py
# CKPT=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/tsm_imagenet-pretrained-mobilenetv2_attnlast_8xb16-1x1x8-50e_kinetics400-rgb-dist/uni2tsm-attnlast-100epoch/20230703_032656/ckpts/epoch_100.pth
# VIDEO_DIR=/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val/
# VIDEOS=$(ls ${VIDEO_DIR})
# LAYER_NAME=backbone/conv2/activate
# DIR_NAME=last_conv
# mkdir -p tools/visualizations/cam_outputs/${DIR_NAME}
# for i in ${VIDEOS}; do
#     echo $i
#     VIDEO=${VIDEO_DIR}${i}
#     OUTPUT=tools/visualizations/cam_outputs/${DIR_NAME}/${i}.gif
#     python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name ${LAYER_NAME} --out-filename ${OUTPUT} # > tools/vis_cam.log
# done
# OUTPUT=tools/visualizations/cam_outputs/__lt03EF4ao_layer6.gif
# python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name 'backbone/layer2/1/conv/2/conv' --out-filename ${OUTPUT} > tools/vis_cam.log
# --use-frames 

# cloud model
# imgs.shape        torch.Size([1, 96, 3, 224, 224])    [batch_size, num_segments, 3, H, W]
# gradient.shape    torch.Size([197, 96, 768]) -> [1, 768, 8, 14, 14]          source shape: [B, C', Tg, H', W']   target shape: [B, Tg, C', H', W']
# CONFIG=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb_cloud_model_attnlast_edge_video_input_test/stage_2_svd_0.6_finetune/vis_data/config.py
# CKPT=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/raw_model_stage_2_feat_finetune_svd/15_epoch_finetune/ckpt/best_acc_top1_epoch_13.pth
# VIDEO=/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val/_04E5ylO7Vc.mp4
# LAYER_NAME=backbone/transformer/resblocks/11/ln_2
# # LAYER_NAME=backbone/EdgeModel/backbone/layer2/1/conv/2/conv
# OUTPUT=tools/visualizations/cam_outputs/_04E5ylO7Vc.gif
# python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name ${LAYER_NAME} --out-filename ${OUTPUT} > tools/vis_cam.log

# edge model
# imgs.shape        torch.Size([1, 80, 3, 224, 224])    [batch_size, num_segments, 3, H, W]
# gradient.shape    torch.Size([80, 1280, 7, 7])        [B*Tg, C', H', W']
CONFIG=/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast_DAD.py
CKPT=/root/Projects/CENet/work_dirs/tsm_attnlast_DAD/epoch_50.pth

# VIDEO=/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val/_04E5ylO7Vc.mp4
# LAYER_NAME=backbone/conv2/activate
# OUTPUT=tools/visualizations/cam_outputs/_04E5ylO7Vc.gif
# python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name ${LAYER_NAME} --out-filename ${OUTPUT} > tools/vis_cam.log

VIDEO_DIR=/root/autodl-tmp/dataset/DAD/DAD/Tester7/reaching_behind/front_IR
VIDEOS=$(ls ${VIDEO_DIR})
LAYER_NAME=backbone/conv2/activate

# batch cam
# for i in {1..25}
# do
#     # for action in `ls /root/autodl-tmp/dataset/DAD/DAD/Tester10`
#     # for action in 'normal_driving_1' 'normal_driving_2' 'normal_driving_3' 'normal_driving_4' 'normal_driving_5' 'normal_driving_6'
#     for action in 'adjusting_radio' 'drinking' 'messaging_left' 'messaging_right' 'reaching_behind' 'talking_with_passenger' 'talking_with_phone_left' 'talking_with_phone_right'
#     do
#         for img_type in 'front_IR' 'front_depth'
#         do
#             DIR_NAME=last_conv_svd/Tester$i/$action/$img_type
#             VIDEO_DIR=/root/autodl-tmp/dataset/DAD/DAD/Tester$i/$action/$img_type
#             OUTPUT=/root/autodl-tmp/visualizations/cam_outputs/${DIR_NAME}
            
#             mkdir -p $OUTPUT
#             python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO_DIR} \
#                 --target-layer-name ${LAYER_NAME} \
#                 --out-filename ${OUTPUT}/res.gif \
#                 --use-frames # > tools/vis_cam.log
#         done
#     done
# done

# VIDEO_DIR=/root/autodl-tmp/dataset/DAD/DAD/Tester7/talking_with_phone_left/front_IR
# OUTPUT=/root/autodl-tmp/visualizations/cam_outputs/last_conv_svd/Tester7/talking_with_phone_left/front_IR_4

VIDEO_DIR=/root/autodl-tmp/dataset/DAD/DAD/Tester24/adjusting_radio/front_IR
OUTPUT=/root/autodl-tmp/visualizations/cam_outputs/last_conv_svd/Tester24/adjusting_radio/front_IR_2

LAYER_NAME=backbone/conv2/activate
LAYER_NAME=backbone/layer7/0/conv/2/conv

mkdir -p $OUTPUT
python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO_DIR} \
                --target-layer-name ${LAYER_NAME} \
                --out-filename ${OUTPUT}/res.gif \
                --use-frames # > tools/vis_cam.log

# mkdir -p tools/visualizations/cam_outputs/${DIR_NAME}
# for i in ${VIDEOS}; do
#     echo $i
#     VIDEO=${VIDEO_DIR}/${i}
#     OUTPUT=tools/visualizations/cam_outputs/${DIR_NAME}/${i} # .png
#     python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} \
#         --device cpu \
#         --target-layer-name ${LAYER_NAME} \
#         --out-filename ${OUTPUT} \
#         --use-frames # > tools/vis_cam.log
# done

# uniformerv2 tsm inputs finetune
# CONFIG=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_tsm_input_format/finetune/vis_data/config.py
# CKPT=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_tsm_input_format/finetune/ckpt/best_acc_top1_epoch_3.pth
# VIDEO=/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val/_04E5ylO7Vc.mp4
# LAYER_NAME=backbone/transformer/resblocks/11/ln_2
# OUTPUT=tools/visualizations/cam_outputs/_04E5ylO7Vc.gif
# python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name ${LAYER_NAME} --out-filename ${OUTPUT} > tools/vis_cam.log

# pretrained uniformerv2
# CONFIG=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb/20230912_120505/vis_data/config.py
# CKPT=/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/teacher_ckpts/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth
# VIDEO=/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val/_04E5ylO7Vc.mp4
# LAYER_NAME=backbone/transformer/resblocks/11/ln_2
# OUTPUT=tools/visualizations/cam_outputs/_04E5ylO7Vc.gif
# python tools/visualizations/vis_cam.py ${CONFIG} ${CKPT} ${VIDEO} --target-layer-name ${LAYER_NAME} --out-filename ${OUTPUT} > tools/vis_cam.log
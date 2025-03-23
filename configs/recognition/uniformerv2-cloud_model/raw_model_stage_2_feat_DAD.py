_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 8
# TSM format data
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
# model = dict(
#     type='Recognizer3D_Cloud_Model',
#     backbone=dict(
#         type='UniFormerV2_CloudModel',
#         input_resolution=224,
#         patch_size=16,
#         width=768,
#         layers=12,
#         heads=12,
#         t_size=num_frames,
#         dw_reduction=1.5,
#         backbone_drop_path_rate=0.,
#         temporal_downsample=False,
#         no_lmhra=True,
#         double_lmhra=True,
#         return_list=[8, 9, 10, 11],
#         n_layers=4,
#         n_dim=768,
#         n_head=12,
#         mlp_factor=4.,
#         drop_path_rate=0.,
#         mlp_dropout=[0.5, 0.5, 0.5, 0.5],
#         clip_pretrained=True,
#         pretrained='ViT-B/16'),
#     cls_head=dict(
#         type='UniFormerHead',
#         dropout_ratio=0.5,
#         num_classes=400,
#         in_channels=768,
#         average_clips='prob'),
#     # data_preprocessor=dict(
#     #     type='ActionDataPreprocessor',
#     #     mean=[114.75, 114.75, 114.75],
#     #     std=[57.375, 57.375, 57.375],
#     #     format_shape='NCTHW')
#     data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
#     )

model = dict(
    type='Recognizer3D_Cloud_Model',
    backbone=dict(
        type='UniFormerV2_CloudModel_Raw_Stage_2_Feat',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20221219-77d34f81.pth',  # noqa: E501
            prefix='backbone.')),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=400,
        in_channels=768,
        average_clips='prob',
        channel_map=  # noqa: E251
        'configs/recognition/uniformerv2/k710_channel_map/map_k400.json',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20221219-77d34f81.pth',  # noqa: E501
            prefix='cls_head.')),
    # data_preprocessor=dict(
    #     type='ActionDataPreprocessor',
    #     mean=[114.75, 114.75, 114.75],
    #     std=[57.375, 57.375, 57.375],
    #     format_shape='NCTHW')
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    )

# dataset settings
# dataset_type = 'VideoDataset'
# data_root = 'data/kinetics400/videos_train'
# data_root_val = 'data/kinetics400/videos_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

dataset_type = 'VideoDataset'
data_root = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_train'      # 'data/kinetics400/videos_train'
data_root_val = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val'    # 'data/kinetics400/videos_val'
ann_file_train = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/kinetics400_train_list_videos.txt'    # 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/kinetics400_val_list_videos.txt'        # 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/kinetics400_val_list_videos.txt'  

file_client_args = dict(io_backend='disk')

# UniformerV2 format inputs [n, t, c, h, w]
# train_pipeline = [
#     dict(type='DecordInit', **file_client_args),
#     dict(type='UniformSample', clip_len=num_frames, num_clips=1),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(
#         type='PytorchVideoWrapper',
#         op='RandAugment',
#         magnitude=7,
#         num_layers=4),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]

# val_pipeline = [
#     dict(type='DecordInit', **file_client_args),
#     dict(
#         type='UniformSample', clip_len=num_frames, num_clips=1,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]

# test_pipeline = [
#     dict(type='DecordInit', **file_client_args),
#     dict(
#         type='UniformSample', clip_len=num_frames, num_clips=4,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='ThreeCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]

# TSM format inputs [n*t, 3, 224, 224]
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]


train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=105, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# load_from = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/teacher_ckpts/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth'
# load_from = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb_cloud_model/20230801_232438/ckpts/best_acc_top1_epoch_49.pth'
# load_from = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb_cloud_model_attnlast_edge/patch_loss/ckpts/best_acc_top1_epoch_55.pth'
# load_from = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb_cloud_model_attnlast_edge/stm_modified/ckpts/epoch_102.pth'
# 云端 + 蒸馏 top1 acc: 81.58 69.42
# load_from = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb_cloud_model_attnlast_edge/self_distill+optimize_cloud_only_final/ckpts/best_acc_top1_epoch_69.pth'
# resume = True
edge_cfg = dict(
    edge_model_cfg = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_attnlast_8xb16-1x1x8-50e_kinetics400-rgb.py',
    edge_ckpt = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/tsm_imagenet-pretrained-mobilenetv2_attnlast_8xb16-1x1x8-50e_kinetics400-rgb-dist/uni2tsm-attnlast-100epoch/20230703_032656/ckpts/epoch_100.pth',
    # 1+qx
    # edge_ckpt = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/work_dirs/tsm_imagenet-pretrained-mobilenetv2_attnlast_8xb16-1x1x8-50e_kinetics400-rgb-dist/uni2tsm-attnlast-1+qx-100epoch/ckpts/best_acc_top1_epoch_74.pth',
    edge_frozen = True,
    edge_test = False,
    edge_optimize = False,
    edge_feat_loss = False,
    loss_dist = True,
    svd_top_ratio=0.6
)

base_lr = 2e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

# base_lr = 2e-6
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
#     paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
#     clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min_ratio=0.1,
        by_epoch=True,
        begin=5,
        end=105,
        convert_to_iter_based=True)
]
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.5,
#         by_epoch=True,
#         begin=0,
#         end=1,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=4,
#         eta_min_ratio=0.5,
#         by_epoch=True,
#         begin=1,
#         end=5,
#         convert_to_iter_based=True)
# ]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)

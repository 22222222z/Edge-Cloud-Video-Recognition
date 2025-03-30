_base_ = ['../../_base_/default_runtime.py']

# pretrained checkpoints for cloud and edge models
checkpoints=dict(
    cloud_ckpt = '/root/Projects/Edge-Cloud-Video-Recognition/pretrained_ckpts/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth',
    edge_ckpt = '/root/Projects/Edge-Cloud-Video-Recognition/work_dirs/edge_model_DAD/best_acc_top1_epoch_4.pth',
)

# edge model
edge_model = dict(
    type='Recognizer2D',
    # student model settings
    backbone=dict(
        type='MobileNetV2TSMEdgeModel',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        pretrained='mmcls://mobilenet_v2',
        return_feats=True
    ),
    cls_head=dict(
        type='TSMEdgeModelHead',
        num_segments=8,
        num_classes=2,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'
    ),
    neck=dict(
        type='DFC',
        top_ratio=0.6
    ),
)
# cloud model settings
num_frames = 8
model = dict(
    type='Recognizer3DECNet',
    backbone=dict(
        type='UniFormerV2ECNet',
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
        clip_pretrained=True,
        pretrained='ViT-B/16'),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=2,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor = dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53], 
        std=[58.395, 57.12, 57.375]
    ),
    # edge model settings
    edge_model=edge_model,
    edge_feat_idx=2, # which level of intermediate feature for data transmission
    # for self distillation during cooperative training
    distill_loss=dict(
        type='RelationDistLoss',
        beta=1.0,
        gamma=1.0,
        tau=1.0,
        reduction='mean'
    ),
    # load pretrained cloud and edge models
    checkpoints=checkpoints
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/root/autodl-tmp/dataset/DAD/DAD'
data_root_val = '/root/autodl-tmp/dataset/DAD/DAD'
ann_file_train = '/root/autodl-tmp/dataset/DAD/DAD/train_anno_file.txt'
ann_file_val = '/root/autodl-tmp/dataset/DAD/DAD/val_anno_file.txt'
ann_file_test = ann_file_val


file_client_args = dict(io_backend='disk')
clip_len = 1
train_pipeline = [
    dict(
        type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # dataset=dict(type="xxx")
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline,
        filename_tmpl = 'img_{:d}.png')
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
        filename_tmpl = 'img_{:d}.png',
        with_offset=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        filename_tmpl = 'img_{:d}.png',
        with_offset=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

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
        end=55,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)

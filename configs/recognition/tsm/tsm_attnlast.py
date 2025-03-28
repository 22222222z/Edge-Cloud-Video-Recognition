_base_ = [
    '../../_base_/models/tsm_mobilenet_v2_attnlast.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
# dataset_type = 'VideoDataset'
# data_root = 'data/kinetics400/videos_train'
# data_root_val = 'data/kinetics400/videos_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

# dataset settings
dataset_type = 'VideoDataset'
# data_root = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_train'      # 'data/kinetics400/videos_train'
# data_root_val = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/videos_val'    # 'data/kinetics400/videos_val'
# ann_file_train = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/kinetics400_train_list_videos.txt'    # 'data/kinetics400/kinetics400_train_list_videos.txt'
# ann_file_val = '/nfs/volume-411-10/weihuapeng/dataset/Kinetics-400/Kinetics-400/kinetics400_val_list_videos.txt'        # 'data/kinetics400/kinetics400_val_list_videos.txt'
# ann_file_test = '/nfs/volume-411-2/weihuapeng/Project/MMAction2/mmaction2/data_list/k400_rawframe/test.csv' 

data_root = '/root/autodl-tmp/dataset/K400/tiny-Kinetics-400'      # 'data/kinetics400/videos_train'
data_root_val = '/root/autodl-tmp/dataset/K400/tiny-Kinetics-400'    # 'data/kinetics400/videos_val'
ann_file_train = '/root/autodl-tmp/dataset/K400/tiny-k400_train.txt'    # 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = '/root/autodl-tmp/dataset/K400/tiny-k400_train.txt'        # 'data/kinetics400/kinetics400_val_list_videos.txt'


file_client_args = dict(io_backend='disk')

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
    batch_size=16,
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
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        # end=50,
        end=75,
        by_epoch=True,
        milestones=[25, 45, 65],
        # milestones=[25, 45],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    # 从uniformerv2里面抄的
    # paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

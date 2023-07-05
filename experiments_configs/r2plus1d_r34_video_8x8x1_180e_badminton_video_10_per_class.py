model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet2Plus1d',
        depth=34,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv2plus1d'),
        norm_cfg=dict(type='BN3d', requires_grad=True, eps=0.001),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        zero_init_residual=False,
        act_cfg=dict(type='ReLU')),
    cls_head=dict(
        type='I3DHead',
        num_classes=18,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained_models/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb_20200826-ab35a529.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'VideoDataset'
data_root = 'badminton_dataset_ncu'
data_root_val = 'badminton_dataset_ncu'
ann_file_train = 'badminton_dataset_ncu_train_labels_10_per_class.txt'
ann_file_val = 'badminton_dataset_ncu_val_labels.txt'
ann_file_test = 'badminton_dataset_ncu_test_labels.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='badminton_dataset_ncu_train_labels_10_per_class.txt',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames', clip_len=8, frame_interval=8,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='VideoDataset',
        ann_file='badminton_dataset_ncu_val_labels.txt',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=8,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        test_mode=True),
    test=dict(
        type='VideoDataset',
        ann_file='badminton_dataset_ncu_val_labels.txt',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=8,
                num_clips=10,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        test_mode=True))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 50
find_unused_parameters = False
work_dir = './experiments/r2plus1d_r34_video_3d_8x8x1_180e_badminton_10_per_class/'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []

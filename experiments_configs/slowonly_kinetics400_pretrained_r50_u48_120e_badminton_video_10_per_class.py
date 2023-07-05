model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=18,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file_train = 'badminton_dataset_ncu_train_labels_10_per_class_data.pkl'
ann_file_val = 'badminton_dataset_ncu_val_labels_data.pkl'
ann_file_test = 'badminton_dataset_ncu_test_labels_data.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 56)),
    dict(type='CenterCrop', crop_size=56),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 56)),
    dict(type='CenterCrop', crop_size=56),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='PoseDataset',
            ann_file='badminton_dataset_ncu_train_labels_10_per_class_data.pkl',
            data_prefix='badminton_dataset_ncu/',
            pipeline=[
                dict(type='UniformSampleFrames', clip_len=48),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(-1, 64)),
                dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
                dict(type='Resize', scale=(48, 48), keep_ratio=False),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                    right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
                dict(
                    type='GeneratePoseTarget',
                    sigma=0.6,
                    use_score=True,
                    with_kp=True,
                    with_limb=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])),
    val=dict(
        type='PoseDataset',
        ann_file='badminton_dataset_ncu_val_labels_data.pkl',
        data_prefix='badminton_dataset_ncu/',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 56)),
            dict(type='CenterCrop', crop_size=56),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='badminton_dataset_ncu_test_labels_data.pkl',
        data_prefix='badminton_dataset_ncu/',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=10,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 56)),
            dict(type='CenterCrop', crop_size=56),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False,
                double=True,
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[9, 11])
total_epochs = 50
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/slowonly_kinetics400_pretrained_r50_u48_120e_badminton_video_10_per_class'
load_from = 'pretrained_models/k400_posec3d-041f49c6.pth'
resume_from = None
find_unused_parameters = True
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []

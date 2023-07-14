_base_ = ['../configs/_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=384,     # change from 768 to 384 as small videomae
        depth=12,
        num_heads=6,       # change from 12 to 6 as small videomae
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=18,      # change from 400 to 18 as small videomae
        in_channels=384,     # change from 768 to 384 as small videomae
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))


## dataset settings
dataset_type = 'VideoDataset'
data_root = 'badminton_dataset_for_classification_ncu_coach_ver_final'
data_root_val = 'badminton_dataset_for_classification_ncu_coach_ver_final'
ann_file_train = 'badminton_dataset_ncu_coach_train_labels.txt'
ann_file_val = 'badminton_dataset_ncu_coach_val_labels.txt'
ann_file_test = 'badminton_dataset_ncu_coach_test_labels.txt'

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='AccMetric')
test_cfg = dict(type='TestLoop')

load_from = 'pre_trained_models/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
work_dir = 'experiments/badminton_timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb'

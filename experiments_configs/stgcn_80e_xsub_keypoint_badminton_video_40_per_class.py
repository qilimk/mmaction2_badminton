model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=18,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'PoseDataset'
ann_file_train = 'badminton_dataset_ncu_train_labels_data.pkl'
ann_file_val = 'badminton_dataset_ncu_val_labels_data.pkl'
ann_file_test = 'badminton_dataset_ncu_test_labels_data.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file='badminton_dataset_ncu_train_labels_data.pkl',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='PoseNormalize'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]),
    val=dict(
        type='PoseDataset',
        ann_file='badminton_dataset_ncu_val_labels_data.pkl',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='PoseNormalize'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='badminton_dataset_ncu_val_labels_data.pkl',
        data_prefix='badminton_dataset_ncu',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='PoseNormalize'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]))
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/stgcn_80e_xsub_keypoint_badminton_video_40_per_class/'
load_from = 'pretrained_models/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []

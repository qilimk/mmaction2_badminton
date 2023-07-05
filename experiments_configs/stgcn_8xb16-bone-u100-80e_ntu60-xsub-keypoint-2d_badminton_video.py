_base_ = 'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'

dataset_type = 'PoseDataset'
ann_file_train = 'badminton_dataset_ncu_coach_train_labels_data.pkl'
ann_file_val = 'badminton_dataset_ncu_coach_val_labels_data.pkl'
ann_file_test = 'badminton_dataset_ncu_coach_test_labels_data.pkl'


train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            split=None, 
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        split=None, 
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        split=None, 
        test_mode=True))

val_evaluator = dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator

load_from = 'pre_trained_models/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth'
work_dir = 'experiments/badminton_stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d'

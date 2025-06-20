_base_ = [
    './base/dataset_6.py', './base/schedule_sgd.py', './base/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.3)

train_dataloader = dict(
    batch_size=32)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]


# model settings
load_from = 'https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s2',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
    ))


val_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  #dict(type='ConfusionMatrix', num_classes = 6),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='ConfusionMatrix', num_classes = 6),
]
lr = optim_wrapper['optimizer']['lr']
batch_size = train_dataloader['batch_size']
work_dir = './experiments/med_classfiy/results/densenet_6_224/' + f'{lr}_{batch_size}'
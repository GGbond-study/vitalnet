_base_ = [
    './base/dataset_vt_6.py', './base/schedule_sgd.py', './base/default_runtime.py'
]

data_preprocessor = dict(
    type = 'VTClsDataPreprocessor',
    num_classes=6,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    table_mean = [0.0, 23.49, 0.0, 0.0, 99.5, 2.63, 54.23, 35.75, 13.22, 68.23],#[age，bmi, 腹痛(0,1), 腹胀(0,1), 5个指标, 大径线]
    table_std = [1.0, 3.86, 1.0, 1.0, 604.75, 16.54, 138.95, 749.409, 29.66, 37.36]
)
 
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001))
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


#pretrained = '/home/user/ZY_Workspace/Django_VTmamba/mmpretrain/experiments/med_classfiy/results/resnet50_598/0.08_32/best_accuracy_top1_epoch_2.pth'
pretrained = '/home/user/ZY_Workspace/Django_VTmamba/mmpretrain/experiments/med_classfiy/last/Moblienets2/best_accuracy_top1_epoch_29.pth'
load_from = pretrained
model = dict(
    type='VTClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s2',
        out_indices=(3, ),
    ),
    neck=dict(type='VT6APooling', Table = True),
    head=dict(
        type='VTClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
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
work_dir = './experiments/med_classfiy/results/resnet50_598/' + f'{lr}_{batch_size}'
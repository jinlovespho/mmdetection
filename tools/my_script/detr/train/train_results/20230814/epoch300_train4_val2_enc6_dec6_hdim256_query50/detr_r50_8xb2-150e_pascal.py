auto_scale_lr = dict(base_batch_size=16)
backend_args = None
data_root = '/mnt/ssd2/dataset/pascal_voc/VOCdevkit/'
dataset_type = 'VOCDataset'
decoder_layers = 6
default_hooks = dict(
    checkpoint=dict(interval=10, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
encoder_layers = 6
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        embed_dims=256,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            bg_cls_weight=0.1,
            class_weight=1.0,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=20,
        type='DETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6,
        return_intermediate=True),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            2048,
        ],
        kernel_size=1,
        norm_cfg=None,
        num_outs=1,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=50,
    positional_encoding=dict(normalize=True, num_feats=128),
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DETR')
num_queries = 50
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=300,
        gamma=0.1,
        milestones=[
            100,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='VOC2012/ImageSets/Main/val.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2012/'),
        data_root='/mnt/ssd2/dataset/pascal_voc/VOCdevkit/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1000,
        600,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_batch_size = 4
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='VOC2012/ImageSets/Main/train.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2012/'),
        data_root='/mnt/ssd2/dataset/pascal_voc/VOCdevkit/',
        filter_cfg=dict(bbox_min_size=32, filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='VOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_batch_size = 2
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='VOC2012/ImageSets/Main/val.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2012/'),
        data_root='/mnt/ssd2/dataset/pascal_voc/VOCdevkit/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        commit=True,
        define_metric_cfg=None,
        init_kwargs=dict(
            name='epoch300_train4_val2_enc6_dec6_hdim256_query50',
            project='LOCAL_detr_hdim_study'),
        log_code_name=None,
        save_dir=
        '/home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/my_script/detr/train/wandb_results/detr',
        type='WandbVisBackend',
        watch_kwargs=None),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            commit=True,
            define_metric_cfg=None,
            init_kwargs=dict(
                name='epoch300_train4_val2_enc6_dec6_hdim256_query50',
                project='LOCAL_detr_hdim_study'),
            log_code_name=None,
            save_dir=
            '/home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/my_script/detr/train/wandb_results/detr',
            type='WandbVisBackend',
            watch_kwargs=None),
    ])
wandb_path = '/home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/my_script/detr/train/wandb_results/detr'
wandb_project_name = 'LOCAL_detr_hdim_study'
wandb_run_name = 'epoch300_train4_val2_enc6_dec6_hdim256_query50'
work_dir = './train_results/20230814/epoch300_train4_val2_enc6_dec6_hdim256_query50'

find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes=19
select_classes=10
model = dict(
    type='CatSegDetector',
    backbone=dict(
        type='CatSegEvaCLIPViT',
        model_name='EVA02-CLIP-B-16',
        pretrained='checkpoints/FineCLIP_coco_vitb16.pt',
        class_names='datasets/incremental_classes/task1_classes.json',
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=2,
        nheads=4,
        hidden_dim=128,
        pooling_size=(2, 2),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        out_indices=[3, 5, 7, 11],  # 输出中间层特征
        cat_seg_indices=[3, 7],
        pad_len=select_classes+1,
        norm_cfg=norm_cfg,
    ),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg
    ),
    
    rpn_head=dict(
        type='RPNHead',
        in_channels=288,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        num_convs=2
    ),

    roi_head=dict(
        type='CatSegMaskRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=(select_classes+1)*256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='CatSegMaskBBoxHead',
            in_channels=(select_classes+1)*256,
            text_dim=512,
            fc_out_channels=512,
            roi_feat_size=7,
            num_classes=num_classes,
            reg_class_agnostic=True,
            learn_bg=True,
            class_embed='datasets/embeddings/FineCLIP_vitb16.pt',
            seen_classes='datasets/incremental_classes/task1_classes.json',
            all_classes='datasets/incremental_classes/task1_classes.json',
            logit_scale=50.0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1,
        ),
    ),
    
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100
        )
    )
)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=16)
dataset_type = 'CocoDatasetOV'
image_size = (384, 384)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        interpolation='bicubic',
        keep_ratio=False),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[0.48145466*255,
              0.4578275*255,
              0.40821073*255],  # SigLIP2: 0.5 * 255
        std=[0.26862954*255,
             0.26130258*255,
             0.27577711*255],   # SigLIP2: 0.5 * 255
        to_rgb=True),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, interpolation='bicubic'),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.48145466 * 255,
                      0.4578275 * 255,
                      0.40821073 * 255],  # SigLIP2: 0.5 * 255
                std=[0.26862954 * 255,
                     0.26130258 * 255,
                     0.27577711 * 255],  # SigLIP2: 0.5 * 255
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=
        '/data1/lzh/datasets/Incremental/train_task_1.json',
        img_prefix='/data1/lzh/datasets/MSCOCO/2017/train2017',
        seen_classes='datasets/incremental_classes/task1_classes.json',
        all_classes='datasets/incremental_classes/task1_classes.json',
        unseen_classes='datasets/incremental_classes/task0_classes.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/data1/lzh/datasets/Incremental/test_task_1.json',
        img_prefix='/data1/lzh/datasets/MSCOCO/2017/val2017',
        seen_classes='datasets/incremental_classes/task1_classes.json',
        all_classes='datasets/incremental_classes/task1_classes.json',
        unseen_classes='datasets/incremental_classes/task0_classes.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/data1/lzh/datasets/Incremental/test_task_1.json',
        img_prefix='/data1/lzh/datasets/MSCOCO/2017/val2017', 
        seen_classes='datasets/incremental_classes/task1_classes.json',
        all_classes='datasets/incremental_classes/task1_classes.json',
        unseen_classes='datasets/incremental_classes/task0_classes.json',
        pipeline=test_pipeline)
)
evaluation = dict(interval=1, metric=['bbox'])
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0, norm_type=2),
)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=1e-7,

    warmup='linear',
    warmup_iters=22376,
    warmup_ratio=0.001,
    )
runner = dict(type='EpochBasedRunner', max_epochs=40)
# fp16 = dict(loss_scale=512.0)
auto_resume = False
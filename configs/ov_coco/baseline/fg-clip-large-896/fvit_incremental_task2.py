find_unused_parameters = True
num_classes = 40
class_weight = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FViT',
    backbone=dict(
        type='FGCLIPViT_interpolate',  # 用于加载模型的类名称
        model_name='fg-clip-large', # 模型名称
        pretrained=None, # Task 2不要重新加载CLIP，通过load_from继承Task 1的backbone
        norm_cfg=norm_cfg,
        out_indices=[6, 10, 14, 23]), # 输出特征图的block索引
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024], # 需要匹配视觉模型的输出通道数
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[3.5, 7, 14, 28, 56]), # [patch_size/4, patch_size/2, pathc_size, 2*patch_size, ...]
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        num_convs=2),
    roi_head=dict(
        type='FViTRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[3.5, 7, 14, 28]), # [patch_size/4, patch_size/2, pathc_size, 2*patch_size]
        bbox_head=dict(
            type='FViTBBoxHead',
            in_channels=256,
            fc_out_channels=768, # 修改为文本嵌入维度
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=class_weight),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            norm_cfg=norm_cfg,
            fixed_temperature=0,
            learned_temperature=50.0,
            vlm_temperature=75.0,
            alpha=0.1,
            beta=0.8,
            class_embed='datasets/embeddings/fg-clip-large.pt', # 文本嵌入路径
            seen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task2_classes.json',
            all_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task12_classes.json',
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1),
        vlm_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=1,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=768, # 修改为文本嵌入维度
            featmap_strides=[14])), # patch_size
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100)))
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'runs/incremental/fg-clip-large-896/train_task1/latest.pth' # 从task1训练好的模型加载权重
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=64)
dataset_type = 'CocoDatasetOV'
image_size = (896, 896) # 模型分辨率
file_client_args = dict(backend='disk')
train_pipeline = [ # 匹配的图片预处理配置
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
test_pipeline = [ # 匹配官方的图片预处理配置
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
                mean=[0.48145466*255,
                    0.4578275*255,
                    0.40821073*255],  # SigLIP2: 0.5 * 255
                std=[0.26862954*255,
                    0.26130258*255,
                    0.27577711*255],   # SigLIP2: 0.5 * 255
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# Incremental -> Ovd4Incremental
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='/hy-tmp/liuzihan/datasets/Ovd4Incremental/0/train_task_2.json',
        img_prefix='/hy-tmp/liuzihan/datasets/MSCOCO/2017/train2017',
        seen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task2_classes.json',
        all_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task12_classes.json',
        unseen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task1_classes.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/hy-tmp/liuzihan/datasets/Ovd4Incremental/0/test_task_12.json',
        img_prefix='/hy-tmp/liuzihan/datasets/MSCOCO/2017/val2017',
        seen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task2_classes.json',
        all_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task12_classes.json',
        unseen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task1_classes.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/hy-tmp/liuzihan/datasets/Ovd4Incremental/0/test_task_12.json',
        img_prefix='/hy-tmp/liuzihan/datasets/MSCOCO/2017/val2017', 
        seen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task2_classes.json',
        all_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task12_classes.json',
        unseen_classes='/hy-tmp/liuzihan/CLIPSelf/F-ViT/datasets/incremental_classes/task1_classes.json',
        pipeline=test_pipeline)
)
evaluation = dict(interval=1, metric=['bbox'])
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=3)
fp16 = dict(loss_scale=512.0)
auto_resume = False


from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn.functional as F
import torch
# from mmcv.ops import RoIAlign
# from mmdet.core import bbox2roi
# from mmdet.models.builder import build_roi_extractor
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

from .cat_seg_loss import SegOrthogonalLoss  # 直接导入类

@DETECTORS.register_module()
class CatSegDetector(TwoStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None,
                 pseudo_iou_thr=0.5,
                 min_keep=512,

                 old_cfg=None,
                 old_ckpt=None,
                 use_feature_routing=False,
                 old_end=0,
    ):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.pseudo_iou_thr = pseudo_iou_thr
        self.min_keep = min_keep

        self.use_feature_routing = use_feature_routing
        self.old_cfg = old_cfg
        self.old_ckpt = old_ckpt
        self.old_end = old_end

        self.old_model = None
        if self.use_feature_routing and (old_cfg is not None) and (old_ckpt is not None):
            self.old_model = self._build_old_model(old_cfg, old_ckpt)

        self.ortho_loss = SegOrthogonalLoss(loss_weight=0.5)

    def _build_old_model(self, cfg_path, ckpt_path):
        cfg = Config.fromfile(cfg_path)

        # 防止 old config 里也写了 old_cfg/old_ckpt 递归
        if isinstance(cfg.model, dict):
            for k in ['old_cfg', 'old_ckpt', 'use_feature_routing', 'old_end',
                      'teacher_cfg', 'teacher_ckpt', 'teacher_score_thr', 'teacher_box_dilation']:
                cfg.model.pop(k, None)

        old_model = build_detector(cfg.model,
                                   train_cfg=cfg.get('train_cfg'),
                                   test_cfg=cfg.get('test_cfg'))

        # 这里 strict=False，避免你之前那种 unexpected key
        print("loading teacher")
        load_checkpoint(old_model, ckpt_path, map_location='cpu', strict=False)
        print("loading teacher done!")

        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad_(False)
        return old_model

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        res_feats = self.backbone(img)
        # cat_seg_feats = res_feats[-2]
        cat_seg_logits = res_feats[-3]
        topk_indices = res_feats[-2]
        if self.with_neck:
            x = self.neck(res_feats[:-3])
        else:
            x = res_feats[:-3]
            

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals


        res = self.roi_head.simple_test(x, proposal_list, img_metas, cat_seg_logits, topk_indices, 
                                        vlm_feat=res_feats[-1],
                                        rescale=rescale)

        return res
    
    def compute_rpn_distillation_loss(self, student_outs, teacher_outs):
        """
        计算 RPN 级的蒸馏损失
        student_outs: Tuple(cls_scores_list, bbox_preds_list)
        teacher_outs: Tuple(cls_scores_list, bbox_preds_list)
        """
        s_cls_scores, s_bbox_preds = student_outs
        t_cls_scores, t_bbox_preds = teacher_outs
        
        loss_rpn_cls = 0.0
        loss_rpn_reg = 0.0
        valid_levels = 0
        
        # 遍历 FPN 的每一层 (通常 5 层)
        for s_cls, s_reg, t_cls, t_reg in zip(s_cls_scores, s_bbox_preds, t_cls_scores, t_bbox_preds):
            # s_cls shape: [Batch, A, H, W] (A=num_anchors)
            # s_reg shape: [Batch, A*4, H, W]
            
            # =================================================================
            # Part 1: Objectness 蒸馏 (让 Student 敢于在旧类位置说"是")
            # =================================================================
            # RPN 的 GT 会把旧类标为背景 (0)。
            # 必须用 MSE 让 Student 逼近 Teacher 的高分 (e.g. 0.9)，抵抗 GT 的 0。
            t_probs = t_cls.sigmoid()
            s_probs = s_cls.sigmoid()
            
            # 全图蒸馏 (MSE)，简单直接
            loss_rpn_cls += F.mse_loss(s_probs, t_probs, reduction='mean')

            # =================================================================
            # Part 2: Regression 蒸馏 (解决你担心的"回归框问题")
            # =================================================================
            # 难点：RPN 输出的是 Offset。背景处的 Offset 是无意义的随机噪声。
            # 策略：只在 Teacher 认为"是物体"的地方蒸馏回归。
            
            # 1. 生成前景掩码 (Mask)
            # 阈值建议 0.7，只选 Teacher 很有把握的区域
            fg_mask = t_probs > 0.7 
            num_pos = fg_mask.sum()
            
            if num_pos > 0:
                # 2. 维度对齐 (这是最容易出错的地方)
                # s_reg 是 [B, A*4, H, W]，我们需要拆出 A 和 4
                B, A4, H, W = s_reg.shape
                A = A4 // 4
                
                # Reshape: [B, A*4, H, W] -> [B, A, 4, H, W]
                # Permute: -> [B, A, H, W, 4] (把坐标维放到最后，方便用 mask 索引)
                s_reg_perm = s_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                t_reg_perm = t_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                
                # 3. 扩展 Mask
                # fg_mask 是 [B, A, H, W]，需要变成 [B, A, H, W, 1] 才能广播到 4 个坐标
                fg_mask_expanded = fg_mask.unsqueeze(-1).expand_as(s_reg_perm)
                
                # 4. 提取有效的前景回归值
                s_pos_reg = s_reg_perm[fg_mask_expanded]
                t_pos_reg = t_reg_perm[fg_mask_expanded]
                
                # 5. 计算 Smooth L1 Loss
                # 相比 MSE，Smooth L1 对回归离群值更不敏感，防止梯度爆炸
                loss_rpn_reg += F.smooth_l1_loss(s_pos_reg, t_pos_reg, reduction='mean')
            
            valid_levels += 1

        # 平均各层的 Loss
        losses = {}
        if valid_levels > 0:
            # 权重建议：
            # RPN 分类很重要，保持 Student 能够产生 Proposal
            losses['loss_rpn_dist_cls'] = (loss_rpn_cls / valid_levels) * 1.0
            
            # RPN 回归是辅助，确保框比较准。权重可以给 1.0 或 0.5
            losses['loss_rpn_dist_reg'] = (loss_rpn_reg / valid_levels) * 1.0
            
        return losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        res_feats = self.backbone(img)
        cat_seg_logits = res_feats[-3]
        topk_indices = res_feats[-2]
        # 4. 多尺度 CLIP 特征 (给 FPN 的原料)
        fpn_inputs = res_feats[:-3]

        if self.with_neck:
            x = self.neck(fpn_inputs)
        else:
            x = fpn_inputs

        x_old = None
        cat_seg_logits_old = None
        teacher_indices = None
        teacher_rpn_outs = None
        teacher_roi_head = None

        if self.use_feature_routing and (self.old_model is not None):
            with torch.no_grad():
                res_feats_old = self.old_model.backbone(img)
                cat_seg_logits_old = res_feats_old[-3]
                teacher_indices = res_feats_old[-2]
                fpn_inputs_old = res_feats_old[:-3]
                x_old = self.old_model.neck(fpn_inputs_old) if self.old_model.with_neck else fpn_inputs_old
                if self.old_model.with_rpn:
                    teacher_rpn_outs = self.old_model.rpn_head(x_old)
                teacher_roi_head = self.old_model.roi_head
        
        losses = dict()

        if cat_seg_logits is not None and cat_seg_logits.shape[1] > 1:
            losses['loss_catseg_ortho'] = self.ortho_loss(cat_seg_logits)

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            # 在原来的多尺度特征图上提取proposal
            rpn_losses, proposal_list = self.rpn_head.forward_train(x,
                                                                    img_metas,
                                                                    gt_bboxes,
                                                                    None,
                                                                    gt_bboxes_ignore,
                                                                    proposal_cfg,
                                                                    **kwargs)
            losses.update(rpn_losses)
            if teacher_rpn_outs is not None:
                student_rpn_outs = self.rpn_head(x)
                # 调用辅助函数计算蒸馏
                loss_rpn_dist = self.compute_rpn_distillation_loss(student_rpn_outs, teacher_rpn_outs)
                losses.update(loss_rpn_dist)
        else:
            proposal_list = proposals

        # num_imgs = len(img_metas)
        # if gt_bboxes_ignore is None:
        #     gt_bboxes_ignore = [None for _ in range(num_imgs)]
        # for i in range(num_imgs):
        #     if gt_bboxes_ignore[i] is None:
        #         # keep on same device
        #         gt_bboxes_ignore[i] = proposal_list[i].new_zeros((0, 4))

        # proposal_list = self._filter_proposals_by_ignore(
        #     proposal_list,
        #     gt_bboxes_ignore,
        #     iou_thr=self.pseudo_iou_thr,
        #     min_keep=self.min_keep
        # )

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 cat_seg_logits, # 传入 Logits
                                                 topk_indices, 

                                                 # 新增传入teacher数据
                                                 x_old=x_old,
                                                 cat_seg_logits_old=cat_seg_logits_old,
                                                 teacher_indices=teacher_indices,
                                                 teacher_roi_head=teacher_roi_head,
                                                 old_end=self.old_end,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

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

    # def _filter_proposals_by_ignore(self, proposal_list, gt_bboxes_ignore,
    #                                iou_thr=0.5, min_keep=512):
    #     """过滤掉与 gt_bboxes_ignore IoU>=iou_thr 的 proposals.
    #     proposal_list: list[Tensor], each (N,4) or (N,5)
    #     gt_bboxes_ignore: list[Tensor], each (M,4)
    #     """
    #     new_list = []
    #     for props, ign in zip(proposal_list, gt_bboxes_ignore):
    #         if props is None or props.numel() == 0:
    #             new_list.append(props)
    #             continue
    #         if ign is None or ign.numel() == 0:
    #             new_list.append(props)
    #             continue

    #         # props 可能是 (N,4) 或 (N,5)，IoU 只用前4维
    #         props_xyxy = props[:, :4].float()
    #         ign_xyxy = ign[:, :4].float()

    #         # bbox_overlaps 支持 CPU/CUDA，跟随输入 device
    #         ious = bbox_overlaps(props_xyxy, ign_xyxy)  # [N, M]
    #         max_iou = ious.max(dim=1)[0] if ious.numel() > 0 else props_xyxy.new_zeros((props_xyxy.size(0),))

    #         keep = max_iou < iou_thr

    #         # 安全：过滤太狠会导致 RCNN sampler 不够样本 -> 回退
    #         if int(keep.sum().item()) < min_keep:
    #             k = min(min_keep, props.size(0))
    #             # 保留“与 ignore 重叠最小”的 k 个
    #             _, idx = torch.topk(max_iou, k=k, largest=False)
    #             new_list.append(props[idx])
    #         else:
    #             new_list.append(props[keep])

    #     return new_list

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
        计算 RPN 蒸馏损失 (使用 Soft BCE)
        student_outs: Tuple(cls_scores, bbox_preds)，每个元素是 List[Tensor] (对应 FPN 层级)
        """
        student_cls_scores = student_outs[0] 
        teacher_cls_scores = teacher_outs[0]
        
        loss_dist = 0.0
        # 蒸馏权重：建议设大一点 (5.0)，因为 RPN 是源头，且 BCE 数值通常较小
        dist_weight = 1.0

        # 遍历 FPN 的每一层 (通常是 5 层)
        for s_cls, t_cls in zip(student_cls_scores, teacher_cls_scores):
            # s_cls shape: [Batch, num_anchors, H, W]
            
            # 1. 获取 Teacher 的软标签 (Sigmoid -> 概率)
            with torch.no_grad():
                t_prob = t_cls.sigmoid()
            
            # 2. 筛选掩码 (Masking)
            # 逻辑：只蒸馏 Teacher 认为是前景 (Object) 的区域 (> 0.1)
            # Teacher 认为是背景的区域，Student 可以自由预测 (可能是新类)
            mask = t_prob > 0.1 
            
            if mask.sum() > 0:
                # 3. 计算 Soft BCE Loss
                # binary_cross_entropy_with_logits 支持 target 为概率值
                s_logits_masked = s_cls[mask]
                t_probs_masked = t_prob[mask]
                
                loss_layer = F.binary_cross_entropy_with_logits(
                    s_logits_masked, 
                    t_probs_masked, 
                    reduction='mean'
                )
                loss_dist += loss_layer
            else:
                pass # 该层没有有效的旧类信号，跳过

        return {'loss_rpn_dist': loss_dist * dist_weight}

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

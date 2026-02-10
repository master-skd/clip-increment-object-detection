from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn.functional as F
import torch
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
        基于 ERD 策略改进的 RPN 蒸馏
        """
        s_cls_scores, s_bbox_preds = student_outs
        t_cls_scores, t_bbox_preds = teacher_outs
        
        # 初始化 Loss Tensor
        loss_rpn_cls = s_cls_scores[0].new_tensor(0.0)
        loss_rpn_reg = s_cls_scores[0].new_tensor(0.0)
        
        valid_levels = 0
        
        # 遍历 FPN 的每一层
        for s_cls, s_reg, t_cls, t_reg in zip(s_cls_scores, s_bbox_preds, t_cls_scores, t_bbox_preds):
            # s_cls shape: [Batch, A, H, W]
            
            # 1. 计算 Teacher 的概率
            t_probs = t_cls.sigmoid()
            s_probs = s_cls.sigmoid()
            
            # =================================================================
            # Part 1: Objectness 蒸馏 (分类)
            # =================================================================
            # 原来是简单的 MSE，这里可以加上简单的空间加权
            # 让 Teacher 认为前景概率高的地方，权重更大
            obj_weight = t_probs.detach() ** 2  # 简单的加权，类似 Focal Loss 的思想
            loss_rpn_cls += (F.mse_loss(s_probs, t_probs, reduction='none') * obj_weight).mean() * 5.0

            # =================================================================
            # Part 2: Regression 蒸馏 (回归) - 应用 ERD
            # =================================================================
            # 计算当前层级的统计量 (Elastic Selection)
            with torch.no_grad():
                batch_mean = t_probs.mean()
                batch_std = t_probs.std()
                
                # 动态阈值：均值 + 0.5倍标准差 (防止太低，设个保底 0.1)
                dynamic_threshold = torch.max(
                    batch_mean + 0.5 * batch_std, 
                    t_probs.new_tensor(0.1)
                )
                
                # 筛选出高价值区域
                fg_mask = t_probs > dynamic_threshold
                
                # 弹性权重：置信度越高，回归 Loss 权重越大
                reg_weight = torch.pow(t_probs, 2.0)
            
            if fg_mask.sum() > 0:
                B, A4, H, W = s_reg.shape
                A = A4 // 4
                
                # 调整回归 tensor 的形状以匹配 mask
                s_reg_perm = s_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                t_reg_perm = t_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                
                # 扩展 mask 和 weight 以匹配回归坐标维度
                # mask: [B, A, H, W] -> [B, A, H, W, 1]
                fg_mask_expanded = fg_mask.unsqueeze(-1).expand_as(s_reg_perm)
                
                # 提取 valid 样本
                s_pos_reg = s_reg_perm[fg_mask_expanded]
                t_pos_reg = t_reg_perm[fg_mask_expanded]
                
                # 提取对应的权重
                # weight: [B, A, H, W] -> [B, A, H, W, 1] -> expand -> masked
                reg_weight_expanded = reg_weight.unsqueeze(-1).expand_as(s_reg_perm)
                weights_pos = reg_weight_expanded[fg_mask_expanded]

                # 计算加权 Smooth L1 Loss
                loss_per_loc = F.smooth_l1_loss(s_pos_reg, t_pos_reg, reduction='none')
                
                # 加权平均
                loss_rpn_reg += (loss_per_loc * weights_pos).sum() / (weights_pos.sum() + 1e-6)

            valid_levels += 1

        if valid_levels > 0:
            losses = {
                'loss_rpn_dist_cls': loss_rpn_cls / valid_levels,
                'loss_rpn_dist_reg': loss_rpn_reg / valid_levels
            }
        else:
            losses = {
                'loss_rpn_dist_cls': loss_rpn_cls,
                'loss_rpn_dist_reg': loss_rpn_reg
            }
            
        return losses

    # def compute_rpn_distillation_loss(self, student_outs, teacher_outs):
    #     """
    #     修正版 RPN 蒸馏：不对称加权，防止抑制新类
    #     """
    #     s_cls_scores, s_bbox_preds = student_outs
    #     t_cls_scores, t_bbox_preds = teacher_outs
        
    #     loss_rpn_cls = s_cls_scores[0].new_tensor(0.0)
    #     loss_rpn_reg = s_cls_scores[0].new_tensor(0.0)
        
    #     valid_levels = 0
        
    #     for s_cls, s_reg, t_cls, t_reg in zip(s_cls_scores, s_bbox_preds, t_cls_scores, t_bbox_preds):
    #         # 1. 概率计算
    #         t_probs = t_cls.sigmoid()
    #         s_probs = s_cls.sigmoid()
            
    #         # =================================================================
    #         # Part 1: Objectness 蒸馏 (分类) - 核心修改 !!!
    #         # =================================================================
            
    #         # 策略：只蒸馏 Teacher 确信的区域。
    #         # 对于 Teacher 认为是背景的区域，给予极低的权重，允许 Student 学习新类。
            
    #         # 1. 基础权重设为极小值 (例如 0.05 或 0.1)
    #         # 这样 Student 在 Teacher 认为是背景的地方可以自由学习 GT
    #         obj_weight = torch.ones_like(t_probs) * 0.1 
            
    #         # 2. 只有 Teacher 确信是物体的地方 (prob > 0.5), 权重才设为高值 (例如 1.0 或 2.0)
    #         # 这保护了旧类不被当成背景
    #         teacher_fg_mask = t_probs > 0.5
    #         obj_weight[teacher_fg_mask] = 2.0
            
    #         # 使用 L1 Loss
    #         diff = F.l1_loss(s_probs, t_probs, reduction='none')
            
    #         # 计算 Loss，这里的系数可以适当降低，比如从 10.0 降到 2.0 或 5.0
    #         # 避免蒸馏 Loss 即使在低权重下也过大
    #         loss_rpn_cls += (diff * obj_weight).mean() * 5.0

    #         # =================================================================
    #         # Part 2: Regression 蒸馏 (保持 ERD 逻辑不变)
    #         # =================================================================
    #         with torch.no_grad():
    #             batch_mean = t_probs.mean()
    #             batch_std = t_probs.std()
    #             dynamic_threshold = torch.max(
    #                 batch_mean + 0.5 * batch_std, 
    #                 t_probs.new_tensor(0.1) 
    #             )
    #             fg_mask = t_probs > dynamic_threshold
    #             reg_weight = torch.pow(t_probs, 2.0)
            
    #         if fg_mask.sum() > 0:
    #             B, A4, H, W = s_reg.shape
    #             A = A4 // 4
    #             s_reg_perm = s_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
    #             t_reg_perm = t_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                
    #             fg_mask_expanded = fg_mask.unsqueeze(-1).expand_as(s_reg_perm)
    #             s_pos_reg = s_reg_perm[fg_mask_expanded]
    #             t_pos_reg = t_reg_perm[fg_mask_expanded]
                
    #             reg_weight_expanded = reg_weight.unsqueeze(-1).expand_as(s_reg_perm)
    #             weights_pos = reg_weight_expanded[fg_mask_expanded]

    #             loss_per_loc = F.smooth_l1_loss(s_pos_reg, t_pos_reg, reduction='none')
    #             loss_rpn_reg += (loss_per_loc * weights_pos).sum() / (weights_pos.sum() + 1e-6)

    #         valid_levels += 1

    #     div = valid_levels if valid_levels > 0 else 1.0
    #     return {
    #         'loss_rpn_dist_cls': loss_rpn_cls / div,
    #         'loss_rpn_dist_reg': loss_rpn_reg / div
    #     }

    def compute_aggregator_distillation_loss(self, s_logits, t_logits):
        # 1. 基础 MSE
        loss_map = F.mse_loss(s_logits, t_logits, reduction='none')
        
        # 2. 激进策略：不要筛选，全图都要对齐！
        # 即使是背景，Student 也要输出和 Teacher 一样的数值 (维持 Response Pattern)
        # 这样能强迫 Aggregator 的参数不发生剧烈漂移
        
        # 为了保留 ERD 的一点思想，我们可以给高分区域加权，但绝不能让低分区域权重为 0
        with torch.no_grad():
            t_probs = t_logits.sigmoid()
            # 基础权重 1.0 + 弹性权重 (让高分区域更重要，但背景也有 1.0 的约束)
            weight = 1.0 + 5.0 * torch.pow(t_probs, 2.0)

        loss_distill = (loss_map * weight).mean()
        
        # 3. 再次加大系数 (Map 蒸馏数值通常很小，需要 x100 甚至更多来抗衡主 Loss)
        return {'loss_agg_dist': loss_distill * 100.0}

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
        teacher_results_bboxes = None

        losses = dict()

        if self.use_feature_routing and (self.old_model is not None):
            with torch.no_grad():
                res_feats_old = self.old_model.backbone(img)
                cat_seg_logits_old = res_feats_old[-3]
                teacher_indices = res_feats_old[-2]
                teacher_text_all = self.old_model.backbone.text_features
                fpn_inputs_old = res_feats_old[:-3]
                x_old = self.old_model.neck(fpn_inputs_old) if self.old_model.with_neck else fpn_inputs_old
                if self.old_model.with_rpn:
                    teacher_rpn_outs = self.old_model.rpn_head(x_old)
                teacher_roi_head = self.old_model.roi_head

                # ! 新增 依据iou剔除rpn中的部分样本
                if self.old_model.with_rpn:
                    proposal_cfg = self.old_model.test_cfg.rpn
                    t_proposals = self.old_model.rpn_head.get_bboxes(
                        *teacher_rpn_outs, img_metas=img_metas, cfg=proposal_cfg)
                else:
                    t_proposals = proposals # 极少情况
                t_roi_outs = self.old_model.roi_head.simple_test(
                    x_old, t_proposals, img_metas, cat_seg_logits_old, teacher_indices,
                    vlm_feat=res_feats_old[-1], rescale=False
                )
                
                # t_roi_outs 通常是 list of numpy array (因为 simple_test 默认最后会转 numpy)
                # 如果你的 simple_test 内部没有转 numpy，那就更好。
                # 如果转了，我们需要在这里转回来
                teacher_results_bboxes = []
                for t_res in t_roi_outs:
                    # t_res 可能是 tuple (bbox, segm) 或者 list of classes
                    # 假设是标准 mmdet 格式：list of np.array (per class)
                    if isinstance(t_res, list) or isinstance(t_res, tuple):
                         # 将所有类别的框拼起来 [N, 5]
                         all_boxes = []
                         # mmdet simple_test 返回的是 list of np.array (长度为 num_classes)
                         # 甚至可能是 tuple (bbox_results, mask_results)
                         bbox_res = t_res[0] if isinstance(t_res, tuple) else t_res
                         
                         if isinstance(bbox_res, list):
                             import numpy as np
                             bbox_res = np.vstack(bbox_res) if len(bbox_res) > 0 else np.zeros((0, 5))
                         
                         teacher_results_bboxes.append(torch.from_numpy(bbox_res).to(img.device).float())
                    else:
                        # 如果已经是 Tensor
                        teacher_results_bboxes.append(t_res)

            # ! === Aggregator Logits 蒸馏 ===
            student_logits_on_old = self.backbone.forward_with_text(
                img,
                teacher_text_all,
                force_indices=teacher_indices
            )

            loss_agg_dist = self.compute_aggregator_distillation_loss(
                student_logits_on_old,
                cat_seg_logits_old
            )
            losses.update(loss_agg_dist)
            
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
                                                                    teacher_results=teacher_results_bboxes,
                                                                    **kwargs)
            losses.update(rpn_losses)
            if teacher_rpn_outs is not None:
                student_rpn_outs = self.rpn_head(x)
                # 调用辅助函数计算蒸馏
                loss_rpn_dist = self.compute_rpn_distillation_loss(student_rpn_outs, teacher_rpn_outs)
                losses.update(loss_rpn_dist)
        else:
            proposal_list = proposals

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

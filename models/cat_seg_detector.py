from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn.functional as F
import torch
from torch import nn
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.models.builder import build_backbone

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
                 history_tasks=None,
                 pretrained=None,
                 init_cfg=None,

                 anchor_weight=0.0,
                 anchor_from_history_idx=-1
                #  pseudo_iou_thr=0.5,
                #  min_keep=512,

                #  old_cfg=None,
                #  old_ckpt=None,
                #  use_feature_routing=False,
                #  old_end=0,
    ):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)

        self.anchor_weight = anchor_weight
        self.anchor_from_history_idx = anchor_from_history_idx
        self.old_neck_params = {}
        
        self.history_backbones = nn.ModuleList()
        if history_tasks is not None and len(history_tasks) > 0:
            print(f"==> [CatSegDetector] Reading {len(history_tasks)} history configs...")
            anchor_idx = anchor_from_history_idx % len(history_tasks)
            for idx, task_info in enumerate(history_tasks):
                old_cfg = Config.fromfile(task_info['config_path'])
                old_model = build_detector(
                    old_cfg.model,
                    train_cfg=old_cfg.get('train_cfg'),
                    test_cfg=old_cfg.get('test_cfg')
                )
                load_checkpoint(old_model, task_info['weight_path'], map_location='cpu', strict=False)
                old_bb = old_model.backbone
                old_bb.eval()
                for param in old_bb.parameters():
                    param.requires_grad = False
                self.history_backbones.append(old_bb)

                if self.anchor_weight > 0 and idx == anchor_idx:
                    print(f"==> [CatSegDetector] Using history task {idx} for neck anchor")
                    for name, p in old_model.neck.named_parameters():
                        # 存成和当前模型 self.named_parameters() 一致的名字
                        self.old_neck_params[f'neck.{name}'] = p.detach().clone().cpu()

                    print("==> [CatSegDetector] Saved old neck params for anchor:")
                    for k in self.old_neck_params.keys():
                        print(f"   {k}")
                # ============================================

                del old_model
            print("==> [CatSegDetector] History backbones loaded seamlessly via mmcv.load_checkpoint!")

        self.ortho_loss = SegOrthogonalLoss(loss_weight=0.5)

    def compute_neck_anchor_loss(self):
        if self.anchor_weight <= 0 or len(self.old_neck_params) == 0:
            return None

        loss_anchor = 0.0
        count = 0

        for name, p in self.named_parameters():
            if not name.startswith('neck.'):
                continue
            if name not in self.old_neck_params:
                continue
            if not p.requires_grad:
                continue

            old_p = self.old_neck_params[name].to(device=p.device, dtype=p.dtype)
            loss_anchor = loss_anchor + F.mse_loss(p, old_p, reduction='sum')
            count += p.numel()

        if count == 0:
            return None

        loss_anchor = loss_anchor / count
        return self.anchor_weight * loss_anchor

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        all_logits = []
        if len(self.history_backbones) > 0:
            with torch.no_grad():
                for old_bb in self.history_backbones:
                    old_bb.eval()
                    old_res_feats = old_bb(img)
                    old_logits = old_res_feats[-2] # 提取历史 Logits
                    all_logits.append(old_logits)

        res_feats = self.backbone(img)
        # cat_seg_logits = res_feats[-2]
        current_logits = res_feats[-2]

        all_logits.append(current_logits)
        cat_seg_logits = torch.cat(all_logits, dim=1)  # 在通道维度拼接 Logits
        # topk_indices = res_feats[-2]

        if self.with_neck:
            x = self.neck(res_feats[:-2])
        else:
            x = res_feats[:-2]

        spatial_attn, _ = torch.max(cat_seg_logits.sigmoid(), dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_attn = spatial_attn.detach()  # 不反向传播到 backbone
        routed_x = []
        for feat in x:
            attn_resized = F.interpolate(spatial_attn, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            routed_feat = feat * (1.0 + attn_resized)
            routed_x.append(routed_feat)
        routed_x = tuple(routed_x)
            
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(routed_x, img_metas)
        else:
            proposal_list = proposals

        res = self.roi_head.simple_test(x, proposal_list, img_metas, cat_seg_logits, 
                                        vlm_feat=res_feats[-1],
                                        rescale=rescale)

        return res
    
    def compute_rpn_distillation_loss_previous(self, student_outs, teacher_outs):
        s_cls_scores, s_bbox_preds = student_outs
        t_cls_scores, t_bbox_preds = teacher_outs

        loss_rpn_cls = s_cls_scores[0].new_tensor(0.0)
        loss_rpn_reg = s_cls_scores[0].new_tensor(0.0)
        valid_levels = 0

        for lvl, (s_cls, s_reg, t_cls, t_reg) in enumerate(zip(s_cls_scores, s_bbox_preds, t_cls_scores, t_bbox_preds)):

            # =========================
            # (1) CLS distill: ERD-FG only
            # =========================
            t_probs = t_cls.sigmoid().detach()
            s_probs = s_cls.sigmoid()

            # ! 原先
            with torch.no_grad():
                mean = t_probs.mean()
                std  = t_probs.std(unbiased=False)
                lam = 0.5
                thr = mean + lam * std
                fg_mask = (t_probs > thr)

                gamma = 2.0
                w = (t_probs.clamp(min=1e-6) ** gamma)

            # 修改
            # with torch.no_grad():
            #     fg_mask_cls, thr_cls = _mask_by_quantile_per_image(t_probs, q=q_cls)
            #     w_cls = (t_probs ** 2)
            #     w_sum = (w_cls * fg_mask_cls.float()).sum().clamp_min(1e-6)

            # ! 原先
            if fg_mask.any():
                diff = F.smooth_l1_loss(s_probs, t_probs, reduction='none')
                loss_fg = (diff[fg_mask] * w[fg_mask]).sum() / (w[fg_mask].sum() + 1e-6)
            else:
                loss_fg = s_probs.sum() * 0.0

            # 修改
            # if fg_mask_cls.any():
            #     diff = F.smooth_l1_loss(s_probs, t_probs, reduction='none')
            #     loss_fg = (diff * w_cls * fg_mask_cls.float()).sum() / w_sum
            # else:
            #     loss_fg = s_probs.sum() * 0.0

            # loss_rpn_cls += loss_fg

            # ! 原先
            # optional very weak bg regularizer
            bg_reg = 0.0
            # bg_reg = s_probs.mean() * 0.01

            loss_rpn_cls += (loss_fg + bg_reg) * 1.0

            # =========================
            # (2) REG distill: keep your ERD logic
            # =========================
            # ! 原先
            with torch.no_grad():
                batch_mean = t_probs.mean()
                batch_std  = t_probs.std(unbiased=False)
                dynamic_threshold = torch.max(batch_mean + 0.5 * batch_std, t_probs.new_tensor(0.1))
                fg_mask_reg = t_probs > dynamic_threshold
                reg_weight = torch.pow(t_probs, 2.0)

            # ! 修改
            # with torch.no_grad():
            #     fg_mask_reg, thr_reg = _mask_by_quantile_per_image(t_probs, q=q_reg)
            #     reg_weight = (t_probs ** 2)

            if fg_mask_reg.sum() > 0:
                B, A4, H, W = s_reg.shape
                A = A4 // 4
                s_reg_perm = s_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)
                t_reg_perm = t_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)

                fg_mask_expanded = fg_mask_reg.unsqueeze(-1).expand_as(s_reg_perm)
                s_pos_reg = s_reg_perm[fg_mask_expanded]
                t_pos_reg = t_reg_perm[fg_mask_expanded]

                reg_weight_expanded = reg_weight.unsqueeze(-1).expand_as(s_reg_perm)
                weights_pos = reg_weight_expanded[fg_mask_expanded]

                loss_per_loc = F.smooth_l1_loss(s_pos_reg, t_pos_reg, reduction='none')
                loss_rpn_reg += (loss_per_loc * weights_pos).sum() / (weights_pos.sum() + 1e-6)

            valid_levels += 1

        # =========================
        # debug (低频)
        # =========================
        # if self.training and (torch.rand(1).item() < 0.1):
        #     fg_ratio_cls = fg_mask_cls.float().mean().item()
        #     fg_ratio_reg = fg_mask_reg.float().mean().item()
        #     print(
        #         f"[RPN-Q] lvl{lvl} "
        #         f"cls_ratio={fg_ratio_cls:.3f} reg_ratio={fg_ratio_reg:.3f} "
        #         f"thr_cls(mean)={thr_cls.mean().item():.3f} thr_reg(mean)={thr_reg.mean().item():.3f} "
        #         f"p(mean)={t_probs.mean().item():.3f} p(std)={t_probs.std(unbiased=False).item():.3f}"
        #     )

        div = valid_levels if valid_levels > 0 else 1.0
        return {
            'loss_rpn_dist_cls': loss_rpn_cls / div,
            'loss_rpn_dist_reg': loss_rpn_reg / div
        }

    def compute_rpn_distillation_loss(self, student_outs, teacher_outs):
        s_cls_scores, s_bbox_preds = student_outs
        t_cls_scores, t_bbox_preds = teacher_outs

        loss_rpn_cls = s_cls_scores[0].new_tensor(0.0)
        loss_rpn_reg = s_cls_scores[0].new_tensor(0.0)
        valid_levels = 0

        T = 1.0          
        lam = 0.5        
        # 分类保底阈值：不需要太高，保护旧类的召回
        thr_floor_cls = 0.15 
        # 回归保底阈值：必须高！绝不能学 Teacher 瞎猜的框
        thr_floor_reg = 0.45 
        
        gamma_cls = 2.0  
        gamma_reg = 2.0  

        for lvl, (s_cls, s_reg, t_cls, t_reg) in enumerate(zip(
            s_cls_scores, s_bbox_preds, t_cls_scores, t_bbox_preds
        )):
            with torch.no_grad():
                t_prob = torch.sigmoid(t_cls / T)  
                B = t_prob.shape[0]
                flat = t_prob.reshape(B, -1)       

                mean = flat.mean(dim=1, keepdim=True)                     
                std  = flat.std(dim=1, unbiased=False, keepdim=True)      
                
                # 1. 分类 Mask (相对宽松)
                thr_cls = torch.maximum(mean + lam * std, flat.new_full((B, 1), thr_floor_cls))
                fg_mask_cls = (flat > thr_cls).reshape_as(t_prob)
                w_cls = (t_prob.clamp_min(1e-6) ** gamma_cls)

                # 2. 回归 Mask (极其严格)
                thr_reg = torch.maximum(mean + lam * std, flat.new_full((B, 1), thr_floor_reg))
                fg_mask_reg = (flat > thr_reg).reshape_as(t_prob)
                w_reg = (t_prob.clamp_min(1e-6) ** gamma_reg)

            # =========================
            # (1) CLS distill: 仅对 Teacher 认为是前景的地方蒸馏，避开新类冲突
            # =========================
            if fg_mask_cls.any():
                with torch.no_grad():
                    t_target = t_prob  

                bce = F.binary_cross_entropy_with_logits(
                    s_cls / T, t_target, reduction='none'
                )
                num = (bce * w_cls * fg_mask_cls.float()).sum()
                den = (w_cls * fg_mask_cls.float()).sum().clamp_min(1e-6)
                loss_fg = (num / den) * (T * T)
            else:
                loss_fg = s_cls.sum() * 0.0

            loss_rpn_cls += loss_fg

            # =========================
            # (2) REG distill: 仅对 Teacher 极其确信的高质量框蒸馏
            # =========================
            if fg_mask_reg.any():
                B, A4, H, W = s_reg.shape
                A = A4 // 4

                s_reg_perm = s_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)  
                t_reg_perm = t_reg.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2)

                m = fg_mask_reg.unsqueeze(-1).expand_as(s_reg_perm)             
                w = w_reg.unsqueeze(-1).expand_as(s_reg_perm)              

                s_pos = s_reg_perm[m]
                t_pos = t_reg_perm[m]
                w_pos = w[m]

                diff = F.smooth_l1_loss(s_pos, t_pos, reduction='none')
                loss_rpn_reg += (diff * w_pos).sum() / (w_pos.sum().clamp_min(1e-6))
            else:
                loss_rpn_reg += s_reg.sum() * 0.0

            valid_levels += 1

        div = valid_levels if valid_levels > 0 else 1.0
        return {
            'loss_rpn_dist_cls': loss_rpn_cls / div,
            'loss_rpn_dist_reg': loss_rpn_reg / div,
        }

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
        cat_seg_logits = res_feats[-2]
        # topk_indices = res_feats[-2]
        # 4. 多尺度 CLIP 特征 (给 FPN 的原料)
        fpn_inputs = res_feats[:-2]

        if self.with_neck:
            x = self.neck(fpn_inputs)
        else:
            x = fpn_inputs

        # x_old = None
        # cat_seg_logits_old = None
        # teacher_indices = None
        # teacher_rpn_outs = None
        # teacher_roi_head = None

        losses = dict()

        # if self.use_feature_routing and (self.old_model is not None):
        #     with torch.no_grad():
        #         res_feats_old = self.old_model.backbone(img)
        #         cat_seg_logits_old = res_feats_old[-3]
        #         teacher_indices = res_feats_old[-2]
        #         teacher_text_all = self.old_model.backbone.text_features
        #         fpn_inputs_old = res_feats_old[:-3]
        #         x_old = self.old_model.neck(fpn_inputs_old) if self.old_model.with_neck else fpn_inputs_old
        #         if self.old_model.with_rpn:
        #             teacher_rpn_outs = self.old_model.rpn_head(x_old)
        #         teacher_roi_head = self.old_model.roi_head

                # ! 新增 依据iou剔除rpn中的部分样本
                # if self.old_model.with_rpn:
                #     proposal_cfg = self.old_model.test_cfg.rpn
                #     t_proposals = self.old_model.rpn_head.get_bboxes(
                #         *teacher_rpn_outs, img_metas=img_metas, cfg=proposal_cfg)
                # else:
                #     t_proposals = proposals # 极少情况

                # ! 修改
                # t_roi_outs = self.old_model.roi_head.simple_test(
                #     x_old, t_proposals, img_metas, cat_seg_logits_old, teacher_indices,
                #     vlm_feat=res_feats_old[-1], rescale=False
                # )
                
                # # t_roi_outs 通常是 list of numpy array (因为 simple_test 默认最后会转 numpy)
                # # 如果你的 simple_test 内部没有转 numpy，那就更好。
                # # 如果转了，我们需要在这里转回来
                # teacher_results_bboxes = []
                # for t_res in t_roi_outs:
                #     # t_res 可能是 tuple (bbox, segm) 或者 list of classes
                #     # 假设是标准 mmdet 格式：list of np.array (per class)
                #     if isinstance(t_res, list) or isinstance(t_res, tuple):
                #          # 将所有类别的框拼起来 [N, 5]
                #          all_boxes = []
                #          # mmdet simple_test 返回的是 list of np.array (长度为 num_classes)
                #          # 甚至可能是 tuple (bbox_results, mask_results)
                #          bbox_res = t_res[0] if isinstance(t_res, tuple) else t_res
                         
                #          if isinstance(bbox_res, list):
                #              import numpy as np
                #              bbox_res = np.vstack(bbox_res) if len(bbox_res) > 0 else np.zeros((0, 5))
                         
                #          teacher_results_bboxes.append(torch.from_numpy(bbox_res).to(img.device).float())
                #     else:
                #         # 如果已经是 Tensor
                #         teacher_results_bboxes.append(t_res)

                # 1) 用 teacher 的 roi_head 直接得到 det_bboxes / det_labels（不走 bbox2result）
                # det_bboxes, det_labels = self.old_model.roi_head.simple_test_bboxes(
                #     x_old,
                #     img_metas,
                #     t_proposals,
                #     self.old_model.test_cfg.rcnn,
                #     cat_seg_feats=cat_seg_logits_old,
                #     topk_indices=teacher_indices,
                #     vlm_feat=res_feats_old[-1],
                #     rescale=False
                # )
                # # 2) 组装 old-only 的 teacher_results_bboxes（每张图一个 [Ni,5]）
                # teacher_results_bboxes = []
                # score_thr=0.3
                # for bboxes_i, labels_i in zip(det_bboxes, det_labels):
                #     if bboxes_i.numel() == 0:
                #         teacher_results_bboxes.append(bboxes_i.new_zeros((0, 5)))
                #         continue

                #     old_mask = (labels_i < self.old_end) & (bboxes_i[:, 4] > score_thr)
                #     teacher_results_bboxes.append(bboxes_i[old_mask])

            # ! === Aggregator Logits 蒸馏 ===
            # student_logits_on_old = self.backbone.forward_with_text(
            #     img,
            #     teacher_text_all,
            #     force_indices=teacher_indices
            # )

            # loss_agg_dist = self.compute_aggregator_distillation_loss(
            #     student_logits_on_old,
            #     cat_seg_logits_old
            # )
            # losses.update(loss_agg_dist)
            
        if cat_seg_logits is not None and cat_seg_logits.shape[1] > 1:
            losses['loss_catseg_ortho'] = self.ortho_loss(cat_seg_logits)

        spatial_attn, _ = torch.max(cat_seg_logits.sigmoid(), dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_attn = spatial_attn.detach()  # 不反向传播到 backbone
        routed_x = []
        for feat in x:
            attn_resized = F.interpolate(spatial_attn, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            routed_feat = feat * (1.0 + attn_resized)
            routed_x.append(routed_feat)
        routed_x = tuple(routed_x)

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            # 在原来的多尺度特征图上提取proposal
            rpn_losses, proposal_list = self.rpn_head.forward_train(routed_x,
                                                                    img_metas,
                                                                    gt_bboxes,
                                                                    None,
                                                                    gt_bboxes_ignore,
                                                                    proposal_cfg,
                                                                    **kwargs)
            losses.update(rpn_losses)
            # if teacher_rpn_outs is not None:
            #     student_rpn_outs = self.rpn_head(x)
            #     # 调用辅助函数计算蒸馏
            #     loss_rpn_dist = self.compute_rpn_distillation_loss(student_rpn_outs, teacher_rpn_outs)
            #     losses.update(loss_rpn_dist)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 cat_seg_logits, # 传入 Logits
                                                #  topk_indices, 

                                                 # 新增传入teacher数据
                                                #  x_old=x_old,
                                                #  cat_seg_logits_old=cat_seg_logits_old,
                                                #  teacher_indices=teacher_indices,
                                                #  teacher_roi_head=teacher_roi_head,
                                                #  old_end=self.old_end,
                                                 **kwargs)
        losses.update(roi_losses)

        loss_neck_anchor = self.compute_neck_anchor_loss()
        if loss_neck_anchor is not None:
            losses['loss_neck_anchor'] = loss_neck_anchor

        return losses

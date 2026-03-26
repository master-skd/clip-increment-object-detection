from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn.functional as F
import torch
import torch.utils.checkpoint as cp
from einops import rearrange

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

@DETECTORS.register_module()
class FViTDist(TwoStageDetector):

    ###############################################################
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,

                 old_cfg=None,
                 old_ckpt=None,
                 old_end=0,
                 init_cfg=None):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)

        self.old_cfg = old_cfg
        self.old_ckpt = old_ckpt
        self.old_end = old_end
        self.old_model = None
        if (old_cfg is not None) and (old_ckpt is not None):
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # num_seen_classes = self.roi_head.bbox_head.num_classes
        # num_classes = self.roi_head.bbox_head.all_embed.shape[1] - 1
        # self.roi_head.bbox_head.num_classes = num_classes
        # self.roi_head.mask_head.num_classes = num_classes
        assert self.with_bbox, 'Bbox head must be implemented.'
        mlvl_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(mlvl_feats[:-1])
        else:
            x = mlvl_feats[:-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        res = self.roi_head.simple_test(x,
                                        proposal_list,
                                        img_metas,
                                        vlm_feat=mlvl_feats[-1],
                                        rescale=rescale)

        return res
    
    def compute_rpn_distillation_loss(self, student_outs, teacher_outs):
        # s_cls_scores, s_bbox_preds = student_outs
        # # --- 强行静音 RPN 蒸馏，保持计算图不断 ---
        # return {
        #     'loss_rpn_dist_cls': s_cls_scores[0].sum() * 0.0,
        #     'loss_rpn_dist_reg': s_bbox_preds[0].sum() * 0.0
        # }
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
                      gt_captions=None,
                      gt_embeds=None,
                      **kwargs):
        # 多尺度图像
        res_feats = self.backbone(img)
        # FPN
        if self.with_neck:
            x = self.neck(res_feats[:-1])
        else:
            x = res_feats[:-1]

        losses = dict()

        x_old = None
        teacher_rpn_outs = None
        teacher_roi_head = None
        res_feats_old = None

        if (self.old_model is not None):
            with torch.no_grad():
                res_feats_old = self.old_model.backbone(img)
                x_old = self.old_model.neck(res_feats_old[:-1]) if self.old_model.with_neck else res_feats_old[:-1]
                if self.old_model.with_rpn:
                    teacher_rpn_outs = self.old_model.rpn_head(x_old)
                teacher_roi_head = self.old_model.roi_head

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                                                      x,
                                                      img_metas,
                                                      gt_bboxes,
                                                      None,
                                                      gt_bboxes_ignore,
                                                      proposal_cfg,
                                                      **kwargs
                                                      )
            losses.update(rpn_losses)
            if teacher_rpn_outs is not None:
                student_rpn_outs = self.rpn_head(x)
                loss_rpn_dist = self.compute_rpn_distillation_loss(student_rpn_outs, teacher_rpn_outs)
                losses.update(loss_rpn_dist)
        else:
            proposal_list = proposals
        # 使用lambda包装函数，将关键字参数转换为位置参数
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 x_old=x_old,
                                                 teacher_roi_head=teacher_roi_head,
                                                 teacher_vlm_feats=(res_feats_old[-1] if res_feats_old else None),
                                                 old_end=self.old_end,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

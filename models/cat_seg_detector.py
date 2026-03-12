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
                 ewc_weight=1.0,
                 fisher_path=None,
                 pretrained=None,
                 init_cfg=None,
    ):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.history_backbones = nn.ModuleList()
        self.ewc_weight = ewc_weight
        self.ewc_enable = False
        self.ewc_param_names = []
        if history_tasks is not None and len(history_tasks) > 0:
            print(f"==> [CatSegDetector] Reading {len(history_tasks)} history configs...")
            fisher_dict = torch.load(fisher_path, map_location='cpu') if fisher_path else None
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

                # EWC初始化
                is_last_task = (idx == len(history_tasks) - 1)
                if is_last_task and fisher_dict is not None:
                    print("==> [EWC] Initializing Elastic Weight Consolidation from the latest task...")
                    old_state_dict = old_model.state_dict()
                    protected_prefixes = ['neck.']
                    for name, param in self.named_parameters():
                        if any(name.startswith(p) for p in protected_prefixes) and param.requires_grad:
                            buf_name_old = 'ewc_old_' + name.replace('.', '_')
                            buf_name_fisher = 'ewc_fisher_' + name.replace('.', '_')
                            self.register_buffer(buf_name_old, old_state_dict[name].clone().detach())
                            self.register_buffer(buf_name_fisher, fisher_dict[name].clone().detach())
                            self.ewc_param_names.append((name, buf_name_old, buf_name_fisher))
                    self.ewc_enable = True
                    print(f"==> [EWC] Protection enabled for {len(self.ewc_param_names)} core feature tensors.")

                del old_model
            print("==> [CatSegDetector] History backbones loaded seamlessly via mmcv.load_checkpoint!")

        self.ortho_loss = SegOrthogonalLoss(loss_weight=1.0)

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

        if self.with_neck:
            x = self.neck(res_feats[:-2])
        else:
            x = res_feats[:-2]

        losses = dict()

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
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 cat_seg_logits, # 传入 Logits
                                                 **kwargs)
        losses.update(roi_losses)

        # EWC
        if self.ewc_enable and self.training:
            ewc_loss = 0.0
            current_params = dict(self.named_parameters())
            for name, buf_name_old, buf_name_fisher in self.ewc_param_names:
                current_p = current_params[name]
                old_p = getattr(self, buf_name_old)
                fisher_p = getattr(self, buf_name_fisher)
                ewc_loss += (fisher_p * (current_p - old_p).pow(2)).sum()
            losses['loss_ewc'] = ewc_loss * (self.ewc_weight / 2.0)

        return losses

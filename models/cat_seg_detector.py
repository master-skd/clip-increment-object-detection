from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.nn.functional as F
import torch
from torch import nn
import copy
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
                #  history_tasks=None,
                 ewc_weight=1000.0,
                 fisher_path=None,
                 pretrained=None,
                 init_cfg=None,
                 prev_model_path=None,
                 old_end=19,
                 use_gda_fpn=False,
                 gda_weight=1.0,
                 gda_lambda=1.0,
                 gda_eps=1e-12,
                 gda_ref_source='loss',
                 gda_loss_key_includes=('loss_roi_dist', 'loss_dist', 'loss_old'),
                 gda_fallback_to_feat=False,
                 use_nsgp_fpn=False,
                 nsgp_proj_path=None,
                 nsgp_lambda=1.0
    ):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.bg_gate = nn.Parameter(torch.tensor([1.0]))
        self.history_backbones = nn.ModuleList()
        self.ewc_weight = ewc_weight
        self.ewc_enable = False
        self.ewc_param_names = []
        self.use_gda_fpn = use_gda_fpn
        self.gda_weight = gda_weight
        self.gda_lambda = gda_lambda
        self.gda_eps = gda_eps
        self.gda_ref_source = gda_ref_source
        self.gda_loss_key_includes = tuple(gda_loss_key_includes)
        self.gda_fallback_to_feat = gda_fallback_to_feat
        self.gda_enable = False
        self.gda_neck_param_names = []
        self.gda_old_neck = None
        self._gda_ref_grads = {}
        self._gda_hook_handles = []
        self._gda_warned_no_ref_loss = False
        self.use_nsgp_fpn = use_nsgp_fpn
        self.nsgp_proj_path = nsgp_proj_path
        self.nsgp_lambda = nsgp_lambda
        self.nsgp_enable = False
        self.nsgp_neck_param_names = []
        self._nsgp_transforms = {}
        self._nsgp_hook_handles = []
        # if history_tasks is not None and len(history_tasks) > 0:
        #     print(f"==> [CatSegDetector] Reading {len(history_tasks)} history configs...")
        #     fisher_dict = torch.load(fisher_path, map_location='cpu') if fisher_path else None
        #     for idx, task_info in enumerate(history_tasks):
        #         old_cfg = Config.fromfile(task_info['config_path'])
        #         old_model = build_detector(
        #             old_cfg.model,
        #             train_cfg=old_cfg.get('train_cfg'),
        #             test_cfg=old_cfg.get('test_cfg')
        #         )
        #         load_checkpoint(old_model, task_info['weight_path'], map_location='cpu', strict=False)
        #         old_bb = old_model.backbone
        #         old_bb.eval()
        #         for param in old_bb.parameters():
        #             param.requires_grad = False
        #         self.history_backbones.append(old_bb)

        #         # EWC初始化
        #         is_last_task = (idx == len(history_tasks) - 1)
        #         if is_last_task and fisher_dict is not None:
        #             print("==> [EWC] Initializing Elastic Weight Consolidation from the latest task...")
        #             old_state_dict = old_model.state_dict()
        #             protected_prefixes = ['neck.']
        #             for name, param in self.named_parameters():
        #                 if any(name.startswith(p) for p in protected_prefixes) and param.requires_grad:
        #                     buf_name_old = 'ewc_old_' + name.replace('.', '_')
        #                     buf_name_fisher = 'ewc_fisher_' + name.replace('.', '_')
        #                     self.register_buffer(buf_name_old, old_state_dict[name].clone().detach())
        #                     self.register_buffer(buf_name_fisher, fisher_dict[name].clone().detach())
        #                     self.ewc_param_names.append((name, buf_name_old, buf_name_fisher))
        #             self.ewc_enable = True
        #             print(f"==> [EWC] Protection enabled for {len(self.ewc_param_names)} core feature tensors.")

        #         del old_model
        #     print("==> [CatSegDetector] History backbones loaded seamlessly via mmcv.load_checkpoint!")

        old_state_dict = None
        if prev_model_path is not None:
            prev_checkpoint = torch.load(prev_model_path, map_location='cpu')
            old_state_dict = prev_checkpoint.get('state_dict', prev_checkpoint)
            old_state_dict = {k.replace('module.', ''): v for k, v in old_state_dict.items()}

        if fisher_path is not None and old_state_dict is not None:
            fisher_dict = torch.load(fisher_path, map_location='cpu')
            fisher_dict = {k.replace('module.', ''): v for k, v in fisher_dict.items()}
            print("==> [EWC] Initializing Elastic Weight Consolidation from the latest task...")

            protected_prefixes = ['neck.']
            for name, param in self.named_parameters():
                if any(name.startswith(p) for p in protected_prefixes) and param.requires_grad:
                    if name not in old_state_dict or name not in fisher_dict:
                        continue
                    buf_name_old = 'ewc_old_' + name.replace('.', '_')
                    buf_name_fisher = 'ewc_fisher_' + name.replace('.', '_')
                    self.register_buffer(buf_name_old, old_state_dict[name].clone().detach())
                    self.register_buffer(buf_name_fisher, fisher_dict[name].clone().detach())
                    self.ewc_param_names.append((name, buf_name_old, buf_name_fisher))
            self.ewc_enable = len(self.ewc_param_names) > 0
            print(f"==> [EWC] Protection enabled for {len(self.ewc_param_names)} core feature tensors.")

        if self.use_gda_fpn and old_state_dict is not None and self.with_neck:
            self.gda_old_neck = copy.deepcopy(self.neck)
            old_neck_state = {}
            for k, v in old_state_dict.items():
                if k.startswith('neck.'):
                    old_neck_state[k[len('neck.'):]] = v
            self.gda_old_neck.load_state_dict(old_neck_state, strict=False)
            self.gda_old_neck.eval()
            for p in self.gda_old_neck.parameters():
                p.requires_grad = False

            for name, param in self.named_parameters():
                if name.startswith('neck.') and param.requires_grad:
                    self.gda_neck_param_names.append(name)
            self.gda_enable = len(self.gda_neck_param_names) > 0
            print(f"==> [GDA] Gradient decomposition enabled for {len(self.gda_neck_param_names)} neck params.")

        if self.use_nsgp_fpn and self.with_neck and self.nsgp_proj_path is not None:
            nsgp_checkpoint = torch.load(self.nsgp_proj_path, map_location='cpu')
            transform_dict = nsgp_checkpoint.get('transforms', nsgp_checkpoint)
            if not isinstance(transform_dict, dict):
                transform_dict = {}

            for name, param in self.named_parameters():
                if not name.startswith('neck.'):
                    continue
                if (not param.requires_grad) or (not name.endswith('.weight')):
                    continue
                transform = transform_dict.get(name, None)
                if transform is None:
                    continue
                self.nsgp_neck_param_names.append(name)
                self._nsgp_transforms[name] = transform.detach().clone().float().cpu()

            self.nsgp_enable = len(self.nsgp_neck_param_names) > 0
            print(f"==> [NSGP] Null-space projection enabled for {len(self.nsgp_neck_param_names)} neck weight params.")

        self.ortho_loss = SegOrthogonalLoss(loss_weight=1.0)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        # all_logits = []
        # if len(self.history_backbones) > 0:
        #     with torch.no_grad():
        #         for old_bb in self.history_backbones:
        #             old_bb.eval()
        #             old_res_feats = old_bb(img)
        #             old_logits = old_res_feats[-2] # 提取历史 Logits
        #             all_logits.append(old_logits)

        res_feats = self.backbone(img)
        cat_seg_logits = res_feats[-2]
        # current_logits = res_feats[-2]

        # all_logits.append(current_logits)
        # cat_seg_logits = torch.cat(all_logits, dim=1)  # 在通道维度拼接 Logits

        if self.with_neck:
            x = self.neck(res_feats[:-2])
        else:
            x = res_feats[:-2]

        spatial_attn, _ = torch.max(cat_seg_logits.sigmoid(), dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_attn = spatial_attn.detach()  # 不反向传播到 backbone
        gate_val = torch.clamp(self.bg_gate, min=0.0)
        routed_x = []
        for feat in x:
            attn_resized = F.interpolate(spatial_attn, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            routed_feat = feat * (gate_val + attn_resized)
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

    def _clear_gda_hooks(self):
        for h in self._gda_hook_handles:
            h.remove()
        self._gda_hook_handles = []
        self._gda_ref_grads = {}

    def _clear_nsgp_hooks(self):
        for h in self._nsgp_hook_handles:
            h.remove()
        self._nsgp_hook_handles = []

    def _clear_reg_hooks(self):
        self._clear_gda_hooks()
        self._clear_nsgp_hooks()

    def _make_gda_hook(self, name):
        def _hook(grad):
            if grad is None:
                return grad
            g_ref = self._gda_ref_grads.get(name, None)
            if g_ref is None:
                return grad
            g_ref = g_ref.to(device=grad.device, dtype=grad.dtype)
            dot = torch.sum(grad * g_ref)
            if dot >= 0:
                return grad
            denom = torch.sum(g_ref * g_ref).clamp_min(self.gda_eps)
            return grad - self.gda_lambda * (dot / denom) * g_ref
        return _hook

    def _prepare_gda_from_ref_loss(self, ref_loss):
        named_params = dict(self.named_parameters())
        neck_params = [named_params[n] for n in self.gda_neck_param_names if n in named_params]
        if len(neck_params) == 0:
            return

        ref_grads = torch.autograd.grad(
            ref_loss,
            neck_params,
            retain_graph=True,
            allow_unused=True
        )

        for n, g in zip(self.gda_neck_param_names, ref_grads):
            if g is not None:
                self._gda_ref_grads[n] = g.detach()

        if len(self._gda_ref_grads) == 0:
            return

        for n in self.gda_neck_param_names:
            if n in self._gda_ref_grads:
                h = named_params[n].register_hook(self._make_gda_hook(n))
                self._gda_hook_handles.append(h)

    def _prepare_gda_for_neck(self, res_feats, neck_feats):
        self._clear_gda_hooks()
        if (not self.gda_enable) or (self.gda_old_neck is None) or (not self.training):
            return

        with torch.no_grad():
            old_neck_feats = self.gda_old_neck(tuple(res_feats[:-2]))

        # g_pseudo: neck feature consistency gradient from old neck supervision.
        gda_proxy_loss = neck_feats[0].new_tensor(0.0)
        for cur_feat, old_feat in zip(neck_feats, old_neck_feats):
            gda_proxy_loss = gda_proxy_loss + F.mse_loss(cur_feat, old_feat, reduction='mean')
        gda_proxy_loss = gda_proxy_loss * self.gda_weight

        self._prepare_gda_from_ref_loss(gda_proxy_loss)

    def _prepare_gda_from_losses(self, losses, res_feats, neck_feats):
        self._clear_gda_hooks()
        if (not self.gda_enable) or (not self.training):
            return

        if self.gda_ref_source == 'feat':
            self._prepare_gda_for_neck(res_feats, neck_feats)
            return

        ref_loss = None
        if self.gda_ref_source in ('loss', 'auto'):
            for k, v in losses.items():
                if not torch.is_tensor(v):
                    continue
                if not v.requires_grad:
                    continue
                if any(token in k for token in self.gda_loss_key_includes):
                    ref_loss = v if ref_loss is None else (ref_loss + v)

        if ref_loss is not None:
            self._prepare_gda_from_ref_loss(ref_loss * self.gda_weight)
            return

        if self.gda_ref_source == 'loss' and (not self._gda_warned_no_ref_loss):
            print("==> [GDA] No matched loss key for loss-based reference gradient. "
                  "Please check gda_loss_key_includes or switch gda_ref_source='feat'/'auto'.")
            self._gda_warned_no_ref_loss = True

        # Optional fallback: feature-level old-neck proxy (or auto mode).
        if (self.gda_ref_source == 'auto' or self.gda_fallback_to_feat) and self.gda_old_neck is not None:
            self._prepare_gda_for_neck(res_feats, neck_feats)

    def _make_nsgp_hook(self, name):
        def _hook(grad):
            if grad is None:
                return grad
            transform = self._nsgp_transforms.get(name, None)
            if transform is None:
                return grad

            transform = transform.to(device=grad.device, dtype=grad.dtype)
            grad_shape = grad.shape
            if grad.ndim == 4:
                grad_flat = grad.reshape(grad_shape[0], -1)
                if transform.shape[0] != grad_flat.shape[1] or transform.shape[1] != grad_flat.shape[1]:
                    return grad
                grad_proj = torch.mm(grad_flat, transform).reshape_as(grad)
            elif grad.ndim == 2:
                if transform.shape[0] != grad_shape[1] or transform.shape[1] != grad_shape[1]:
                    return grad
                grad_proj = torch.mm(grad, transform)
            else:
                return grad

            lam = float(self.nsgp_lambda)
            return grad + lam * (grad_proj - grad)
        return _hook

    def _prepare_nsgp_hooks(self):
        self._clear_nsgp_hooks()
        if (not self.nsgp_enable) or (not self.training):
            return

        named_params = dict(self.named_parameters())
        for name in self.nsgp_neck_param_names:
            if name not in named_params:
                continue
            if name not in self._nsgp_transforms:
                continue
            h = named_params[name].register_hook(self._make_nsgp_hook(name))
            self._nsgp_hook_handles.append(h)

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
        cat_seg_logits = res_feats[-2]  # [B, K, H, W]

        if self.with_neck:
            x = self.neck(res_feats[:-2])
        else:
            x = res_feats[:-2]
        self._clear_reg_hooks()

        losses = dict()

        if cat_seg_logits is not None and cat_seg_logits.shape[1] > 1:
            losses['loss_catseg_ortho'] = self.ortho_loss(cat_seg_logits)

        if self.training and len(self.history_backbones) > 0 and not getattr(self, '_gate_reset_done', False):
            print("\n==> [Gate] New Task Detected! Forcefully resetting bg_gate to 1.0 to break cold-start deadlock!")
            self.bg_gate.data.fill_(1.0)
            self._gate_reset_done = True # 标记已开闸，之后再也不执行

        spatial_attn, _ = torch.max(cat_seg_logits.sigmoid(), dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_attn = spatial_attn.detach()
        gate_val = torch.clamp(self.bg_gate, min=0.0)
        routed_x = []
        for feat in x:
            attn_resized = F.interpolate(spatial_attn, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            routed_feat = feat * (gate_val + attn_resized)
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
        if self.nsgp_enable and self.training:
            self._prepare_nsgp_hooks()
        if self.gda_enable and self.training:
            self._prepare_gda_from_losses(losses, res_feats, x)

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

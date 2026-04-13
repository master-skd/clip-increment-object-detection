from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.utils.checkpoint as cp
import torch
from einops import rearrange


@DETECTORS.register_module()
class FViT(TwoStageDetector):

    ###############################################################
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 ewc_weight=1000.0,
                 fisher_path=None,
                 pretrained=None,
                 init_cfg=None,
                 prev_model_path=None):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.ewc_weight = ewc_weight
        self.ewc_enable = False
        self.ewc_param_names = []

        if fisher_path is not None and prev_model_path is not None:
            fisher_dict = torch.load(fisher_path, map_location='cpu')
            fisher_dict = {k.replace('module.', ''): v for k, v in fisher_dict.items()}
            print("==> [EWC] Initializing Elastic Weight Consolidation from the latest task...")

            prev_checkpoint = torch.load(prev_model_path, map_location='cpu')
            old_state_dict = prev_checkpoint.get('state_dict', prev_checkpoint)
            old_state_dict = {k.replace('module.', ''): v for k, v in old_state_dict.items()}

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
            # rpn_losses, proposal_list = self.rpn_head.forward_train(
            #     x,
            #     img_metas,
            #     gt_bboxes,
            #     gt_labels=None,
            #     gt_bboxes_ignore=gt_bboxes_ignore,
            #     proposal_cfg=proposal_cfg,
            #     **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        # 使用lambda包装函数，将关键字参数转换为位置参数
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 gt_captions=gt_captions,
                                                 gt_embeds=gt_embeds,
                                                 **kwargs)
        losses.update(roi_losses)

        if self.ewc_enable and self.training:
            ewc_loss = x[0].new_tensor(0.0)
            current_params = dict(self.named_parameters())
            for name, buf_name_old, buf_name_fisher in self.ewc_param_names:
                current_p = current_params[name]
                old_p = getattr(self, buf_name_old)
                fisher_p = getattr(self, buf_name_fisher)
                ewc_loss = ewc_loss + (fisher_p * (current_p - old_p).pow(2)).sum()
            losses['loss_ewc'] = ewc_loss * (self.ewc_weight / 2.0)

        return losses

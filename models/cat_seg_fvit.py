from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch.utils.checkpoint as cp
from einops import rearrange
import torch.nn.functional as F
import mmcv
from mmcv.visualization import imshow_det_bboxes
import os
import numpy as np
import json

@DETECTORS.register_module()
class CatSegFViT(TwoStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.gt_ann_path = '/hy-tmp/liuzihan/datasets/test/test_task_1.json'
        self._gt_cache = None

    def _prepare_gt_cache(self):
        """Load GT annotations once for visualization."""
        if self._gt_cache is not None:
            return
        cache = dict()
        ann_path = getattr(self, 'gt_ann_path', None)
        if ann_path is None or not os.path.exists(ann_path):
            self._gt_cache = cache
            return
        try:
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
        except Exception:
            self._gt_cache = cache
            return

        images = ann_data.get('images', [])
        annotations = ann_data.get('annotations', [])
        id_to_name = {img.get('id'): os.path.basename(img.get('file_name', '')) for img in images}

        for ann in annotations:
            image_id = ann.get('image_id')
            file_name = id_to_name.get(image_id)
            if not file_name:
                continue
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            entry = cache.setdefault(file_name, {'bboxes': [], 'labels': []})
            entry['bboxes'].append([x1, y1, x2, y2])
            entry['labels'].append(ann.get('category_id', -1))

        self._gt_cache = cache

    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, gt_bboxes=None, gt_labels=None):
        assert self.with_bbox, 'Bbox head must be implemented.'
        self._prepare_gt_cache()
        res_feats = self.backbone(img)
        cat_seg_feats = res_feats[-1]
        if self.with_neck:
            x = self.neck(res_feats[:-1])
        else:
            x = res_feats[:-1]

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        res = self.roi_head.simple_test(x, proposal_list, img_metas, cat_seg_feats=cat_seg_feats,
                                        rescale=rescale)

        # 可视化结果
        os.makedirs('show', exist_ok=True)
        
        for i, (result, meta) in enumerate(zip(res, img_metas)):
            img_path = meta.get('filename')
            img_array = mmcv.imread(img_path)
            canvas = img_array.copy()
            
            # 提取预测框和标签
            pred_bboxes = []
            pred_labels = []
            
            for class_idx, class_bboxes in enumerate(result):
                if len(class_bboxes) > 0:
                    for bbox in class_bboxes:
                        pred_bboxes.append(bbox)
                        pred_labels.append(class_idx)
            
            if len(pred_bboxes) > 0:
                pred_bboxes = np.array(pred_bboxes, dtype=np.float32)
                pred_labels = np.array(pred_labels, dtype=np.int32)
            else:
                pred_bboxes = np.zeros((0, 5), dtype=np.float32)
                pred_labels = np.array([], dtype=np.int32)
            
            canvas = imshow_det_bboxes(
                canvas,
                pred_bboxes,
                pred_labels,
                class_names=None,
                show=False,
                out_file=None,
                wait_time=0
            )
            
            # 如果有GT信息，叠加GT框
            cached_gt = None
            if gt_bboxes is not None and gt_labels is not None:
                cached_gt = (gt_bboxes[i], gt_labels[i])
            elif self._gt_cache:
                file_key = os.path.basename(img_path)
                cache_entry = self._gt_cache.get(file_key)
                if cache_entry:
                    cached_gt = (
                        np.array(cache_entry['bboxes'], dtype=np.float32),
                        np.array(cache_entry['labels'], dtype=np.int32))

            if cached_gt is not None:
                gt_box, gt_label = cached_gt
                if len(gt_box) > 0:
                    gt_box_np = gt_box.cpu().numpy() if hasattr(gt_box, 'cpu') else gt_box
                    gt_box_with_score = np.concatenate([
                        gt_box_np,
                        np.ones((len(gt_box_np), 1))
                    ], axis=1)
                    gt_label_np = gt_label.cpu().numpy() if hasattr(gt_label, 'cpu') else gt_label
                    canvas = imshow_det_bboxes(
                        canvas,
                        gt_box_with_score,
                        gt_label_np,
                        class_names=None,
                        bbox_color='red',
                        text_color='red',
                        show=False,
                        out_file=None,
                        wait_time=0
                    )
            
            # 构建输出路径
            out_file = os.path.join('show', os.path.basename(img_path))
            mmcv.imwrite(canvas, out_file)

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
        res_feats = self.backbone(img)
        cat_seg_feats = res_feats[-1]
        if self.with_neck:
            x = self.neck(res_feats[:-1])
        else:
            x = res_feats[:-1]
        
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
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 cat_seg_feats=cat_seg_feats,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

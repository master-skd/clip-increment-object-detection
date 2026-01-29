import argparse
import os
import json
import sys
from tqdm import tqdm

import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint, get_dist_info, init_dist

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import multi_gpu_test, single_gpu_test

# 防止 import error
sys.path.append(os.getcwd())
try:
    from models.cat_seg_detector import CatSegDetector
    from datasets.coco_ov import CocoDatasetOV
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Task1 pseudo labels on Task2 images and merge into Task2 annotation JSON (keep/restore full categories).'
    )

    # === 基础路径（按你给的默认值）===
    parser.add_argument('--config', default='configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask.py')
    parser.add_argument('--checkpoint', default='runs/cat-seg/test_train_task1_10_2d_pseudo_label/epoch_40.pth')
    parser.add_argument('--ann-file', default='/mnt/data14/yyg/datasets/Incremental/train_task_2.json')
    parser.add_argument('--img-prefix', default='/mnt/data14/yyg/datasets/MSCOCO/2017/train2017')
    parser.add_argument('--task1-classes', default='datasets/incremental_classes/task1_classes.json')

    # === 输出文件（按你给的默认值）===
    parser.add_argument('--out', default='train_task2_with_pseudo_labels.json',
                        help='Output JSON file name')

    # === 核心阈值（你原来是 0.08；你现在想低一点做过滤，默认改成 0.01 更符合新目的）===
    parser.add_argument('--score-thr', type=float, default=0.01,
                        help='Pseudo label threshold. Recommend low (e.g. 0.01) for filtering old-class regions.')

    # === 防止 ID 冲突（你原来是 10000000）===
    parser.add_argument('--ann-id-start', type=int, default=10000000)

    # === categories 修复：默认不传则用 ann-file 的 categories；需要强制时再传 source-categories ===
    parser.add_argument('--source-categories', default=None,
                        help='Optional: json file whose categories are guaranteed correct. If set, output categories will be copied from it.')

    # === dataloader（按你给的默认值）===
    parser.add_argument('--samples-per-gpu', type=int, default=8)
    parser.add_argument('--workers-per-gpu', type=int, default=4)

    # === 分布式参数（按你给的默认值）===
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_categories_from_source(source_path):
    with open(source_path, 'r') as f:
        data = json.load(f)
    cats = data.get('categories', [])
    return cats


def main():
    args = parse_args()

    # 1) init dist
    cfg = Config.fromfile(args.config)

    if args.launcher == 'pytorch' and 'RANK' not in os.environ:
        print("⚠️  Distributed env not detected. Falling back to single-GPU mode.")
        args.launcher = 'none'

    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', dict(backend='nccl')))
        rank, _ = get_dist_info()

    # 2) load task1 classes
    if rank == 0:
        print(f"Loading Task1 classes from: {args.task1_classes}")
    with open(args.task1_classes, 'r') as f:
        task1_classes = json.load(f)
    assert isinstance(task1_classes, list) and len(task1_classes) > 0

    # 3) read task2 json as source of truth (for annotations/categories)
    if rank == 0:
        print(f"Reading Task2 json (source-of-truth) from: {args.ann_file}")
    with open(args.ann_file, 'r') as f:
        task2_data = json.load(f)

    # choose categories for output
    if args.source_categories is not None:
        correct_categories = load_categories_from_source(args.source_categories)
        if rank == 0:
            print(f"[Categories] Using categories from --source-categories: {args.source_categories} (N={len(correct_categories)})")
    else:
        correct_categories = task2_data.get('categories', [])
        if rank == 0:
            print(f"[Categories] Using categories from --ann-file: {args.ann_file} (N={len(correct_categories)})")

    if rank == 0 and len(correct_categories) == 0:
        print("⚠️  WARNING: categories field is empty in source. Output json will have empty categories unless you pass --source-categories.")

    # 4) build dataset for inference (use task1 classes, but task2 images)
    # IMPORTANT: do NOT use dataset.CLASSES to build categories later!
    cfg.data.test.classes = task1_classes
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.img_prefix = args.img_prefix
    cfg.data.test.test_mode = True
    cfg.data.test.filter_empty_gt = False

    # model class alignment
    if hasattr(cfg.model.roi_head.bbox_head, 'all_classes'):
        cfg.model.roi_head.bbox_head.all_classes = args.task1_classes
    if hasattr(cfg.model.roi_head.bbox_head, 'seen_classes'):
        cfg.model.roi_head.bbox_head.seen_classes = args.task1_classes
    if hasattr(cfg.model.backbone, 'class_json_path'):
        cfg.model.backbone.class_json_path = args.task1_classes

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # 5) build model for inference
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 6) inference
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=True)

    # 7) merge (rank0)
    if rank != 0:
        return

    print(f"\nProcessing {len(outputs)} results...")
    merged_annotations = list(task2_data.get('annotations', []))
    print(f"Loaded original GT annotations: {len(merged_annotations)}")

    ann_id_counter = int(args.ann_id_start)
    pseudo_count = 0

    # Build a fast image_id mapping if dataset.data_infos order differs (usually aligned)
    # Here we assume outputs order == dataset order.
    for idx, img_result in enumerate(tqdm(outputs, desc="Generating Pseudo Labels")):
        img_info = dataset.data_infos[idx]
        img_id = img_info['id']

        # img_result: list over classes (task1 only), each is [Ni, 5] (x1,y1,x2,y2,score)
        for label_idx, bboxes in enumerate(img_result):
            if bboxes is None or getattr(bboxes, 'shape', None) is None or bboxes.shape[0] == 0:
                continue

            valid = bboxes[:, 4] > args.score_thr
            b = bboxes[valid]
            if b.shape[0] == 0:
                continue

            for box in b:
                x1, y1, x2, y2, score = box.tolist()
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                new_ann = {
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": int(label_idx),  # task1 label id (0..18)  <-- 你要的旧类
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                    # "ignore": True,
                    "score": float(score),
                    "is_pseudo": True
                }
                merged_annotations.append(new_ann)
                ann_id_counter += 1
                pseudo_count += 1

    print(f"Total generated pseudo labels: {pseudo_count} (thr={args.score_thr})")
    print(f"Total combined annotations: {len(merged_annotations)}")

    out_json = {
        "images": task2_data.get("images", []),
        "annotations": merged_annotations,
        "categories": correct_categories
    }

    # 保留原文件里可能存在的其他字段（可选）
    # for k in task2_data.keys():
    #     if k not in out_json:
    #         out_json[k] = task2_data[k]

    print(f"Saving to: {args.out}")
    with open(args.out, 'w') as f:
        json.dump(out_json, f)
    print("✅ Done.")


if __name__ == '__main__':
    main()

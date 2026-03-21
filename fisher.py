import torch
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.utils import Config
from tqdm import tqdm
import os

from models.cat_seg_detector import CatSegDetector
from datasets.coco_ov import CocoDatasetOV

def compute_fisher(config_path, checkpoint_path, save_path, old_fisher_path=None, alpha=0.9, num_samples=500):
    print("==> [Fisher] Loading model and checkpoint...")
    cfg = Config.fromfile(config_path)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.cuda()

    model.train()
    protected_prefixs = ['neck.']
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in protected_prefixs):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("==> [Fisher] Building dataset...")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=True
    )

    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    print(f"==> [Fisher] Start computing Empirical Fisher for {num_samples} samples...")

    model.zero_grad()
    processed = 0
    for i, data in enumerate(tqdm(data_loader)):
        if processed >= num_samples:
            break
        
        data['img'] = data['img'].data[0].cuda()
        data['img_metas'] = data['img_metas'].data[0]
        data['gt_bboxes'] = [b.cuda() for b in data['gt_bboxes'].data[0]]
        data['gt_labels'] = [l.cuda() for l in data['gt_labels'].data[0]]
        if 'gt_masks' in data:
            data['gt_masks'] = data['gt_masks'].data[0]
        
        # 前向传播计算 Loss
        losses = model(return_loss=True, **data)
        
        # 将所有 loss 汇总成一个标量
        loss = sum([loss.sum() for loss in losses.values() if isinstance(loss, torch.Tensor)])
        
        # 清空上一步的梯度，进行反向传播
        model.zero_grad()
        loss.backward()
        
        # 🌟 收集梯度的平方！
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 累加：(grad^2) / num_samples，这就是期望公式！
                fisher_dict[name] += (param.grad.data ** 2) / num_samples
                
        processed += 1

    if old_fisher_path is not None and os.path.exists(old_fisher_path):
        print(f"\n==> [Online EWC] Fusing with historical Fisher matrix from: {old_fisher_path}")
        print(f"==> [Online EWC] Memory decay coefficient (alpha) = {alpha}")
        old_fisher_dict = torch.load(old_fisher_path, map_location='cuda')
        for name in fisher_dict.keys():
            if name in old_fisher_dict:
                fisher_dict[name] = fisher_dict[name] + alpha * old_fisher_dict[name]
        print("==> [Onlien EWC] Fusion complete! The fossil has been upgraded.")

    # 6. 保存为可以直接加载的字典权重文件
    print(f"==> [Fisher] Saving Fisher matrix to {save_path}...")
    # 转回 CPU 保存，防止占 GPU 显存
    fisher_dict_cpu = {k: v.cpu() for k, v in fisher_dict.items()}
    torch.save(fisher_dict_cpu, save_path)
    print("==> [Fisher] Done! 🚀")

if __name__ == '__main__':
    # 示例用法：你可以直接用命令行传参，或者在这里写死
    compute_fisher(
        config_path='configs/ov_coco/cat_seg_2d_4tasks/catseg_mask_task2.py',    # Task 2 的配置文件
        checkpoint_path='runs/cat-seg/test_train_task2_10_2d_moe_sigmoid/epoch_20.pth', # Task 2 训好的权重
        save_path='fisher_task2.pth',       # 你想保存 Fisher 矩阵的位置
        old_fisher_path='fisher_task1.pth',  # Task 1 的 Fisher 矩阵路径，如果没有就设为 None
        alpha=1.0,
        num_samples=500  # 抽样数量
    )

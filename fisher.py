import torch
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.utils import Config
from mmcv.parallel import DataContainer
from tqdm import tqdm
import os
import argparse
import importlib


def _register_local_modules():
    # Ensure custom detectors/heads/datasets are registered into MMDet registries.
    importlib.import_module('models')
    importlib.import_module('datasets')


def _strip_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def _unwrap_data_container(x):
    if isinstance(x, DataContainer):
        if len(x.data) == 0:
            return None
        x = x.data[0]
    if isinstance(x, dict):
        return {k: _unwrap_data_container(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unwrap_data_container(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_unwrap_data_container(v) for v in x)
    return x


def _move_tensors_to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda(non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_tensors_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_tensors_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_tensors_to_cuda(v) for v in x)
    return x


def _prepare_batch(data):
    def _normalize_gt_bboxes(t):
        if not torch.is_tensor(t):
            return t
        if t.ndim == 2 and t.shape[-1] == 4:
            return t
        if t.numel() == 0:
            return t.new_zeros((0, 4))
        if t.ndim == 1 and t.numel() % 4 == 0:
            return t.view(-1, 4)
        return t.reshape(-1, 4)

    def _normalize_gt_labels(t):
        if not torch.is_tensor(t):
            return t
        if t.ndim == 1:
            return t
        if t.numel() == 0:
            return t.new_zeros((0,), dtype=t.dtype)
        return t.reshape(-1)

    batch = {}
    for key, val in data.items():
        val = _unwrap_data_container(val)
        if val is None:
            continue

        # MMDet two-stage detectors expect per-image lists for these fields.
        if key in ('gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_masks'):
            if not isinstance(val, (list, tuple)):
                val = [val]
            moved = []
            for item in val:
                if torch.is_tensor(item):
                    if key == 'gt_bboxes':
                        item = _normalize_gt_bboxes(item)
                    elif key == 'gt_labels':
                        item = _normalize_gt_labels(item)
                    moved.append(item.cuda(non_blocking=True))
                else:
                    moved.append(item)
            batch[key] = moved
            continue

        if key == 'img':
            batch[key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
            continue

        batch[key] = _move_tensors_to_cuda(val)
    return batch


def _sum_loss_tensors(losses):
    loss_tensors = []
    for loss_val in losses.values():
        if torch.is_tensor(loss_val):
            loss_tensors.append(loss_val.mean())
        elif isinstance(loss_val, list):
            for item in loss_val:
                if torch.is_tensor(item):
                    loss_tensors.append(item.mean())
    if len(loss_tensors) == 0:
        return None
    return sum(loss_tensors)


def compute_fisher(
    config_path,
    checkpoint_path,
    save_path,
    old_fisher_path=None,
    alpha=0.9,
    num_samples=500,
    protected_prefixes=('neck.',),
    samples_per_gpu=1,
    workers_per_gpu=2
):
    print("==> [Fisher] Loading model and checkpoint...")
    _register_local_modules()
    cfg = Config.fromfile(config_path)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.cuda()

    model.train()
    protected_prefixes = tuple(protected_prefixes)
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in protected_prefixes):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("==> [Fisher] Building dataset...")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=True
    )

    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    print(f"==> [Fisher] Start computing Empirical Fisher for {num_samples} samples...")

    processed = 0
    for i, data in enumerate(tqdm(data_loader)):
        if processed >= num_samples:
            break

        batch = _prepare_batch(data)

        losses = model(return_loss=True, **batch)
        loss = _sum_loss_tensors(losses)
        if loss is None:
            continue

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)

        processed += 1

    if processed == 0:
        raise RuntimeError("No valid batches were processed when computing Fisher.")
    for name in fisher_dict:
        fisher_dict[name] = fisher_dict[name] / processed

    if old_fisher_path is not None and os.path.exists(old_fisher_path):
        print(f"\n==> [Online EWC] Fusing with historical Fisher matrix from: {old_fisher_path}")
        print(f"==> [Online EWC] Memory decay coefficient (alpha) = {alpha}")
        old_fisher_dict = torch.load(old_fisher_path, map_location='cpu')
        old_fisher_dict = _strip_module_prefix(old_fisher_dict)
        for name in fisher_dict.keys():
            if name in old_fisher_dict:
                fisher_dict[name] = fisher_dict[name] + alpha * old_fisher_dict[name].to(fisher_dict[name].device)
        print("==> [Online EWC] Fusion complete! The fossil has been upgraded.")

    print(f"==> [Fisher] Saving Fisher matrix to {save_path}...")
    fisher_dict_cpu = {k: v.cpu() for k, v in fisher_dict.items()}
    torch.save(fisher_dict_cpu, save_path)
    print("==> [Fisher] Done!")


def compute_fisher_fvit(
    config_path,
    checkpoint_path,
    save_path,
    old_fisher_path=None,
    alpha=0.9,
    num_samples=500,
    samples_per_gpu=1,
    workers_per_gpu=2
):
    return compute_fisher(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        save_path=save_path,
        old_fisher_path=old_fisher_path,
        alpha=alpha,
        num_samples=num_samples,
        protected_prefixes=('neck.',),
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute empirical Fisher for EWC.')
    parser.add_argument('--config', required=True, help='Config path of the previous-task model.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path of the previous-task model.')
    parser.add_argument('--save-path', required=True, help='Where to save fisher .pth.')
    parser.add_argument('--old-fisher-path', default=None, help='Optional previous fisher for online EWC fusion.')
    parser.add_argument('--alpha', type=float, default=0.9, help='Online EWC decay coefficient.')
    parser.add_argument('--num-samples', type=int, default=500, help='How many mini-batches to use.')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='Dataloader samples_per_gpu.')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Dataloader workers_per_gpu.')
    parser.add_argument('--model-type', choices=['fvit', 'generic'], default='fvit',
                        help='Use fvit to compute Fisher for FViT neck parameters.')
    args = parser.parse_args()

    if args.model_type == 'fvit':
        compute_fisher_fvit(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            save_path=args.save_path,
            old_fisher_path=args.old_fisher_path,
            alpha=args.alpha,
            num_samples=args.num_samples,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu
        )
    else:
        compute_fisher(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            save_path=args.save_path,
            old_fisher_path=args.old_fisher_path,
            alpha=args.alpha,
            num_samples=args.num_samples,
            protected_prefixes=('neck.',),
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu
        )

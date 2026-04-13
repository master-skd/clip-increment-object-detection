import argparse
import importlib
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from torch import nn
from tqdm import tqdm


def _register_local_modules():
    # Ensure custom detectors/heads/datasets are registered into MMDet registries.
    importlib.import_module('models')
    importlib.import_module('datasets')


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


def _adaptive_tail_indices(svals: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
    points = svals.detach().float().cpu().numpy()
    if points.ndim != 1 or points.size == 0:
        return torch.zeros_like(svals, dtype=torch.bool)
    if points.size <= 2:
        idx = points.size - 1
    else:
        diff_o1 = points[:-1] - points[1:]
        if diff_o1.size <= 1:
            idx = points.size - 1
        else:
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            idx = int(np.argmax(diff_o2) + int((points.size - diff_o2.size) / 2))
            idx = max(0, min(idx, points.size - 1))

    if -1.0 <= offset <= 1.0:
        idx = min(idx + int(offset * idx), points.size - 1)
        idx = max(0, idx)
    else:
        idx = max(min(idx + int(offset), points.size - 1), 0)

    tail = np.zeros(points.size, dtype=np.int64)
    tail[idx:] = 1
    return torch.as_tensor(tail, dtype=torch.bool, device=svals.device)


def _tail_by_last_singular(svals: torch.Tensor, thres: float) -> torch.Tensor:
    svals = svals.detach().float()
    if svals.numel() == 0:
        return torch.zeros_like(svals, dtype=torch.bool)
    return svals <= svals[-1] * thres


def _tail_by_rank_ratio(svals: torch.Tensor, tail_ratio: float) -> torch.Tensor:
    n = svals.numel()
    if n == 0:
        return torch.zeros_like(svals, dtype=torch.bool)
    k = int(round(n * tail_ratio))
    k = max(1, min(k, n))
    mask = torch.zeros(n, dtype=torch.bool, device=svals.device)
    mask[n - k:] = True
    return mask


def _select_tail_indices(svals: torch.Tensor, mode: str, offset: float, thres: float, tail_ratio: float):
    if mode == 'adaptive':
        return _adaptive_tail_indices(svals, offset=offset)
    if mode == 'last':
        return _tail_by_last_singular(svals, thres=thres)
    if mode == 'ratio':
        return _tail_by_rank_ratio(svals, tail_ratio=tail_ratio)
    raise ValueError(f'Unsupported select mode: {mode}')


class NSGPCovCollector:
    def __init__(self, model, param_name_set, use_batch_mean=True):
        self.model = model
        self.param_name_set = param_name_set
        self.use_batch_mean = use_batch_mean
        self.cov = OrderedDict()
        self.handles = []
        self.module_to_param = {}

    @torch.no_grad()
    def _update_cov(self, name, x):
        c = torch.mm(x.transpose(0, 1), x)
        if name not in self.cov:
            self.cov[name] = c
        else:
            self.cov[name] = self.cov[name] + c

    @torch.no_grad()
    def _hook(self, module, fea_in, fea_out):
        name = self.module_to_param.get(id(module), None)
        if name is None:
            return
        x = fea_in[0]
        if not torch.is_tensor(x):
            return
        if self.use_batch_mean and x.ndim > 0:
            x = torch.mean(x, dim=0, keepdim=True)

        if isinstance(module, nn.Linear):
            x = x.reshape(-1, x.shape[-1])
            self._update_cov(name, x)
            return

        if isinstance(module, nn.Conv2d):
            x = F.unfold(
                x,
                kernel_size=module.kernel_size,
                padding=module.padding,
                stride=module.stride)
            x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
            self._update_cov(name, x)
            return

    def register(self):
        for module_name, module in self.model.named_modules():
            if not hasattr(module, 'weight'):
                continue
            param_name = module_name + '.weight' if module_name else 'weight'
            if param_name not in self.param_name_set:
                continue
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            self.module_to_param[id(module)] = param_name
            self.handles.append(module.register_forward_hook(self._hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.module_to_param = {}


@torch.no_grad()
def gen_nsgp_transforms(
    config_path,
    checkpoint_path,
    save_path,
    num_samples=500,
    protected_prefixes=('neck.',),
    samples_per_gpu=1,
    workers_per_gpu=2,
    select_mode='adaptive',
    offset=0.0,
    last_thres=1.001,
    tail_ratio=0.3,
    use_batch_mean=True,
):
    print('==> [NSGP] Loading config and model...')
    _register_local_modules()
    cfg = Config.fromfile(config_path)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.cuda()
    model.eval()

    protected_prefixes = tuple(protected_prefixes)
    protected_names = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in protected_prefixes) and name.endswith('.weight'):
            if param.ndim in (2, 4):
                protected_names.append(name)
    if len(protected_names) == 0:
        raise RuntimeError(f'No protected weights found for prefixes: {protected_prefixes}')
    print(f'==> [NSGP] Collecting covariance for {len(protected_names)} params.')

    print('==> [NSGP] Building dataset/dataloader...')
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=True,
    )

    collector = NSGPCovCollector(
        model=model,
        param_name_set=set(protected_names),
        use_batch_mean=use_batch_mean)
    collector.register()

    processed = 0
    for data in tqdm(data_loader):
        if processed >= num_samples:
            break
        batch = _prepare_batch(data)
        _ = model(return_loss=True, **batch)
        processed += 1

    collector.clear()
    if processed == 0:
        raise RuntimeError('No valid batches were processed when collecting NSGP covariance.')
    print(f'==> [NSGP] Processed {processed} mini-batches.')

    transforms = OrderedDict()
    stats = OrderedDict()
    for name in protected_names:
        cov = collector.cov.get(name, None)
        if cov is None:
            continue
        # full_matrices=False gives Vh shape [d, d], singular values sorted desc.
        _, svals, v_h = torch.linalg.svd(cov, full_matrices=False)
        v = v_h.transpose(0, 1)

        tail_idx = _select_tail_indices(
            svals=svals,
            mode=select_mode,
            offset=offset,
            thres=last_thres,
            tail_ratio=tail_ratio)
        if tail_idx.sum() == 0:
            tail_idx[-1] = True

        basis = v[:, tail_idx]
        transform = torch.mm(basis, basis.transpose(0, 1))
        transforms[name] = transform.float().cpu()
        stats[name] = {
            'dim': int(transform.shape[0]),
            'tail_rank': int(tail_idx.sum().item()),
            'total_rank': int(tail_idx.numel()),
            'sval_max': float(svals[0].item()) if svals.numel() > 0 else 0.0,
            'sval_min': float(svals[-1].item()) if svals.numel() > 0 else 0.0,
        }

    if len(transforms) == 0:
        raise RuntimeError('No transforms were generated. Please check prefixes/checkpoint/config.')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {
        'transforms': transforms,
        'meta': {
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'num_samples': processed,
            'protected_prefixes': list(protected_prefixes),
            'select_mode': select_mode,
            'offset': offset,
            'last_thres': last_thres,
            'tail_ratio': tail_ratio,
            'use_batch_mean': use_batch_mean,
            'stats': stats,
        }
    }
    torch.save(payload, save_path)
    print(f'==> [NSGP] Saved transforms: {len(transforms)} params -> {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate NSGP transforms for FPN/neck weights.')
    parser.add_argument('--config', required=True, help='Config path of the previous-task model.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path of the previous-task model.')
    parser.add_argument('--save-path', required=True, help='Output .pth path for NSGP transforms.')
    parser.add_argument('--num-samples', type=int, default=500, help='How many mini-batches to collect covariance.')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='Dataloader samples_per_gpu.')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Dataloader workers_per_gpu.')
    parser.add_argument('--protected-prefixes', nargs='+', default=['neck.'],
                        help='Parameter prefixes to apply NSGP (default: neck.).')
    parser.add_argument('--select-mode', choices=['adaptive', 'last', 'ratio'], default='adaptive',
                        help='Null-space basis selection mode.')
    parser.add_argument('--offset', type=float, default=0.0,
                        help='Offset used by adaptive mode.')
    parser.add_argument('--last-thres', type=float, default=1.001,
                        help='Threshold multiplier for `last` mode: s <= s_min * thres.')
    parser.add_argument('--tail-ratio', type=float, default=0.3,
                        help='Tail rank ratio for `ratio` mode.')
    parser.add_argument('--no-batch-mean', action='store_true',
                        help='Disable averaging over batch before covariance accumulation.')
    args = parser.parse_args()

    gen_nsgp_transforms(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        save_path=args.save_path,
        num_samples=args.num_samples,
        protected_prefixes=tuple(args.protected_prefixes),
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        select_mode=args.select_mode,
        offset=args.offset,
        last_thres=args.last_thres,
        tail_ratio=args.tail_ratio,
        use_batch_mean=(not args.no_batch_mean),
    )

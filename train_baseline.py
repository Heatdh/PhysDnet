import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from baselines.aod_net import AODNet
from baselines.ffa_net import FFANet
from baselines.dehazeformer import DehazeFormer
from dataset import STFDehazeDataset, DehazeDataset, DummyDehazeDataset
from visualize import make_vis_grid, save_vis_grid

def load_config(config_path: str) -> dict:
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    cli_map = {
        'epochs': ('training', 'epochs'),
        'batch_size': ('training', 'batch_size'),
        'lr': ('training', 'lr'),
        'weight_decay': ('training', 'weight_decay'),
        'num_workers': ('training', 'num_workers'),
        'save_dir': ('training', 'save_dir'),
        'log_every': ('training', 'log_every'),
        'val_every': ('training', 'val_every'),
    }
    for arg_name, cfg_path in cli_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            section, key = cfg_path
            cfg.setdefault(section, {})[key] = val
    return cfg

def build_datasets(cfg: dict, args: argparse.Namespace):
    if args.dummy:
        img_size = tuple(cfg.get('data', {}).get('crop_size', [256, 256]))
        train_ds = DummyDehazeDataset(length=200, img_size=img_size)
        val_ds = DummyDehazeDataset(length=40, img_size=img_size)
        return train_ds, val_ds

    data_cfg = cfg.get('data', {})
    if args.dataset == 'stf':
        stf_root = data_cfg.get('stf_root', 'data/stf/SeeingThroughFog')
        depth_dir = data_cfg.get('depth_dir', None)
        meta_dir = data_cfg.get('meta_dir', 'data/stf/meta')

        train_ts = data_cfg.get('train_timestamps') or os.path.join(meta_dir, 'train_timestamps.txt')
        val_ts = data_cfg.get('val_timestamps') or os.path.join(meta_dir, 'val_timestamps.txt')

        crop = tuple(data_cfg.get('crop_size', [512, 512]))
        beta = tuple(data_cfg.get('beta_range', [0.005, 0.04]))
        airlight = tuple(data_cfg.get('airlight_range', [0.7, 1.0]))
        max_depth = data_cfg.get('max_depth', 120.0)

        train_weather = data_cfg.get('train_weather_filter', None)
        val_weather = data_cfg.get('val_weather_filter', None)

        train_ds = STFDehazeDataset(
            stf_root=stf_root, depth_dir=depth_dir,
            timestamps_file=train_ts, crop_size=crop,
            beta_range=beta, airlight_range=airlight,
            max_depth=max_depth, augment=data_cfg.get('augment', True),
            weather_filter=train_weather,
        )
        val_ds = STFDehazeDataset(
            stf_root=stf_root, depth_dir=depth_dir,
            timestamps_file=val_ts, crop_size=crop,
            beta_range=beta, airlight_range=airlight,
            max_depth=max_depth, augment=False,
            weather_filter=val_weather,
        )
    return train_ds, val_ds

def build_inputs(batch, in_channels, device):
    """Construct the input tensor based on in_channels."""
    hazy = batch['hazy'].to(device)
    if in_channels == 3:
        return hazy
    sparse = batch['sparse_depth'].to(device)
    if in_channels == 4:
        return torch.cat([hazy, sparse if sparse.dim() == 4 else sparse.unsqueeze(1)], dim=1)
    mask = batch['mask'].to(device)
    if in_channels == 5:
        return torch.cat([hazy, sparse if sparse.dim() == 4 else sparse.unsqueeze(1), mask if mask.dim() == 4 else mask.unsqueeze(1)], dim=1)
    raise ValueError(f"Unsupported in_channels: {in_channels}")

def get_run_name(cfg, args, model):
    wandb_cfg = cfg.get('wandb', {})
    if wandb_cfg.get('run_name'): return wandb_cfg.get('run_name')
    if args.wandb_run: return args.wandb_run
    parts = [args.dataset, args.model, f"ch{args.in_channels}"]
    parts.append(datetime.now().strftime('%m%d_%H%M'))
    return '_'.join(parts)

def init_wandb(cfg, args, model, run_name):
    if args.no_wandb or not HAS_WANDB: return None
    run = wandb.init(
        project=cfg.get('wandb', {}).get('project', 'lidar-dehazing'),
        name=run_name,
        config={**cfg, 'cli_args': vars(args)}
    )
    return run

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / max(1, self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = self.eta_min + (base_lr - self.eta_min) * factor
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

def validate(model, val_loader, in_channels, device):
    model.eval()
    running_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = build_inputs(batch, in_channels, device)
            clear = batch['clear'].to(device)
            out = model(inputs)['preds']
            loss = F.l1_loss(out, clear)
            running_loss += loss.item()
            n += 1
    return {'loss': running_loss / max(n, 1)}

def visualize_epoch(model, val_dataset, in_channels, device, epoch, cfg, save_dir):
    model.eval()
    samples = []
    n_vis = min(8, len(val_dataset))
    with torch.no_grad():
        for idx in range(n_vis):
            batch = val_dataset[idx]
            clear = batch['clear'].unsqueeze(0).to(device)
            inputs = build_inputs({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}, in_channels, device)
            out = model(inputs)['preds']
            hazy = batch['hazy'].unsqueeze(0) # for viz
            
            # Pad the samples dict with dummy values so make_vis_grid doesn't crash
            samples.append({
                'hazy': hazy[0].cpu(),
                'clear': clear[0].cpu(),
                'restored': out[0].cpu(),
                'physics_restored': out[0].cpu(),  # Duplicate restored image
                'transmission': torch.ones_like(clear[0].cpu())[:1, :, :],  # Dummy black/white map
                'sparse_depth': batch['sparse_depth'].cpu(),
                'mask': torch.ones_like(batch['sparse_depth'].cpu()), # Dummy mask
            })
    grid = make_vis_grid(samples)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'vis_epoch_{epoch:03d}.png')
    save_vis_grid(grid, save_path)
    if HAS_WANDB and wandb.run is not None:
        wandb.log({'val/samples': wandb.Image(grid, caption=f'Epoch {epoch}')})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--dataset', type=str, default='stf')
    p.add_argument('--model', type=str, required=True, choices=['aod', 'ffa', 'dehazeformer'],
                   help='Baseline model to train.')
    p.add_argument('--in_channels', type=int, default=3, choices=[3, 4, 5],
                   help='Input channels: 3 (RGB), 4 (RGB+Depth), 5 (RGB+Depth+Mask)')
    p.add_argument('--dummy', action='store_true')
    
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--wandb_run', type=str, default=None)
    return p.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Model: {args.model}, Channels: {args.in_channels}')

    cfg = load_config(args.config) if args.config else {'data': {}, 'training': {}}
    cfg = merge_config_with_args(cfg, args)
    train_cfg = cfg.get('training', {})

    train_ds, val_ds = build_datasets(cfg, args)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get('batch_size', 8), shuffle=True,
                              num_workers=train_cfg.get('num_workers', 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get('batch_size', 8), shuffle=False,
                            num_workers=train_cfg.get('num_workers', 4), pin_memory=True)

    if args.model == 'aod':
        model = AODNet(in_channels=args.in_channels).to(device)
    else:
        model = FFANet(in_channels=args.in_channels).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.2f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr', 2e-4), weight_decay=train_cfg.get('weight_decay', 1e-4))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=train_cfg.get('epochs', 100))

    run_name = get_run_name(cfg, args, model)
    run_dir = os.path.join(train_cfg.get('runs_dir', 'runs'), run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    init_wandb(cfg, args, model, run_name)

    best_val_loss = float('inf')

    for epoch in range(1, train_cfg.get('epochs', 100) + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            inputs = build_inputs(batch, args.in_channels, device)
            clear = batch['clear'].to(device)
            
            optimizer.zero_grad()
            out = model(inputs)['preds']
            loss = F.l1_loss(out, clear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if i % train_cfg.get('log_every', 20) == 0:
                print(f'Epoch {epoch} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}')
                if HAS_WANDB and wandb.run:
                    wandb.log({'train/loss': loss.item(), 'train/lr': optimizer.param_groups[0]['lr']})

        scheduler.step()
        train_loss /= len(train_loader)
        
        val_metrics = validate(model, val_loader, args.in_channels, device)
        val_loss = val_metrics['loss']
        print(f'Epoch {epoch} time: {time.time()-t0:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if HAS_WANDB and wandb.run:
            wandb.log({'val/loss': val_loss, 'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(run_dir, 'best.pth'))
            print('  -> Best model saved.')

        visualize_epoch(model, val_ds, args.in_channels, device, epoch, cfg, os.path.join(run_dir, 'visualizations'))

    if HAS_WANDB and wandb.run: wandb.finish()

if __name__ == '__main__':
    train()


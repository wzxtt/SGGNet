import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model.sggnet import SGGNet
from dataset import OCTAFAZDataset


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='mean')

        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return self.bce_weight * bce + self.dice_weight * dice


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_score(pred, target, smooth=1.0):
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(preds, masks).item() * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(preds, masks).item() * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


def main():
    parser = argparse.ArgumentParser(description='SGGNet Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--img_size', type=int, default=304)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        args.epochs = cfg.get('epochs', args.epochs)
        args.lr = cfg.get('lr', args.lr)
        args.batch_size = cfg.get('batch_size', args.batch_size)
        args.img_size = cfg.get('img_size', args.img_size)
        args.val_split = cfg.get('val_split', args.val_split)
        args.weight_decay = cfg.get('weight_decay', 1e-4)
        args.scheduler = cfg.get('scheduler', 'none')
    else:
        args.weight_decay = 1e-4
        args.scheduler = 'none'

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = OCTAFAZDataset(args.data_root, split='train', img_size=args.img_size)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = SGGNet(in_channels=3, num_classes=1).to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if str(args.scheduler).lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_dir = os.path.join('logs', f'seed_{args.seed}')
    writer = SummaryWriter(log_dir)

    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    best_dice = 0.0

    print(f"Training SGGNet with seed={args.seed}")
    print(f"Train: {train_size}, Val: {val_size}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'seed': args.seed,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Saved best model (Dice: {val_dice:.4f})")

    writer.close()
    print(f"Training complete. Best Val Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()

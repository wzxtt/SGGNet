import os
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.sggnet import SGGNet
from dataset import OCTAFAZDataset


def dice_coefficient(pred, target, smooth=1.0):
    p = (pred > 0.5).float().view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def jaccard_index(pred, target, smooth=1.0):
    p = (pred > 0.5).float().view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return (inter + smooth) / (p.sum() + t.sum() - inter + smooth)


def binary_f1(pred, target):
    p = (pred > 0.5).float().view(-1)
    t = target.view(-1)
    tp = (p * t).sum()
    fp = p.sum() - tp
    fn = t.sum() - tp
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def contour_points(mask):
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    dilated = cv2.dilate(m, kernel, iterations=1)
    contour = dilated - eroded
    pts = np.argwhere(contour > 0).astype(np.float64)
    return pts


def hausdorff_dist(pred_pts, target_pts):
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf')
    from scipy.spatial.distance import directed_hausdorff
    return max(directed_hausdorff(pred_pts, target_pts)[0],
               directed_hausdorff(target_pts, pred_pts)[0])


def hausdorff_95(pred_pts, target_pts):
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf')
    from scipy.spatial.distance import cdist
    d1 = cdist(pred_pts, target_pts)
    d2 = cdist(target_pts, pred_pts)
    return max(np.percentile(d1.max(axis=1), 95),
               np.percentile(d2.max(axis=1), 95))


def avg_surface_dist(pred_pts, target_pts):
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf')
    from scipy.spatial.distance import cdist
    d1 = cdist(pred_pts, target_pts).min(axis=1).mean()
    d2 = cdist(target_pts, pred_pts).min(axis=1).mean()
    return (d1 + d2) / 2.0


def compute_surface_metrics(pred_mask, target_mask):
    try:
        import cv2
        pc = contour_points(pred_mask)
        tc = contour_points(target_mask)
        return hausdorff_dist(pc, tc), hausdorff_95(pc, tc), avg_surface_dist(pc, tc)
    except Exception:
        return float('inf'), float('inf'), float('inf')


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    metrics = {k: [] for k in ['Dice', 'Jaccard', 'BF1', 'HD', 'HD95', 'ASSD']}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = (model(images) > 0.5).float()

        for i in range(images.size(0)):
            metrics['Dice'].append(dice_coefficient(preds[i], masks[i]).item())
            metrics['Jaccard'].append(jaccard_index(preds[i], masks[i]).item())
            metrics['BF1'].append(binary_f1(preds[i], masks[i]).item())
            hd, hd95, assd = compute_surface_metrics(
                preds[i, 0].cpu().numpy(), masks[i, 0].cpu().numpy())
            metrics['HD'].append(hd)
            metrics['HD95'].append(hd95)
            metrics['ASSD'].append(assd)

    results = {}
    for k, v in metrics.items():
        valid = [x for x in v if x != float('inf')]
        results[k] = (np.mean(valid), np.std(valid)) if valid else (0.0, 0.0)
    return results


def main():
    parser = argparse.ArgumentParser(description='SGGNet Evaluation')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

    for seed in args.seeds:
        ckpt_path = os.path.join(args.checkpoint, f'seed_{seed}', 'best_model.pth')
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\nEvaluating seed {seed}...")
        model = SGGNet(in_channels=3, num_classes=1).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        dataset = OCTAFAZDataset(args.data_root, split='test', img_size=args.img_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        results = evaluate_model(model, loader, device)
        print(f"  Seed {seed}:")
        for m, (mean, std) in results.items():
            print(f"    {m}: {mean:.4f} +/- {std:.4f}")

        row = {'seed': seed}
        row.update({k: v[0] for k, v in results.items()})
        all_results.append(row)

    if all_results:
        metric_names = ['Dice', 'Jaccard', 'HD', 'HD95', 'ASSD', 'BF1']
        means = {m: np.mean([r[m] for r in all_results]) for m in metric_names}
        stds = {m: np.std([r[m] for r in all_results]) for m in metric_names}

        print(f"\n{'='*50}")
        print("Overall (mean +/- std across seeds):")
        for m in metric_names:
            print(f"  {m}: {means[m]:.4f} +/- {stds[m]:.4f}")

        csv_path = os.path.join(args.checkpoint, 'results.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Seed'] + metric_names)
            for r in all_results:
                w.writerow([r['seed']] + [f"{r[m]:.4f}" for m in metric_names])
            w.writerow(['Mean'] + [f"{means[m]:.4f}" for m in metric_names])
            w.writerow(['Std'] + [f"{stds[m]:.4f}" for m in metric_names])
        print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()

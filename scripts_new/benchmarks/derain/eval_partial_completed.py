#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
import torchvision.transforms as transforms
import pyiqa


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate already-completed derain outputs only.')
    parser.add_argument('--dataset_root', default='dataset/rain', help='Root folder containing original datasets')
    parser.add_argument('--benchmark_root', default='outputs_derain_benchmark_full', help='Root folder containing restored outputs')
    parser.add_argument('--datasets', nargs='+', default=['Rain100H', 'rain100H_train'], help='Dataset names')
    parser.add_argument('--out_suffix', default='partial', help='Suffix for output files, e.g., partial')
    return parser.parse_args()


class MetricSuite:
    def __init__(self, device: str):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.vif = pyiqa.create_metric('vif', device=device)
        self.fsim = pyiqa.create_metric('fsim', device=device)
        self.niqe = pyiqa.create_metric('niqe', device=device)

    def evaluate(self, pred_path: Path, gt_path: Path) -> Dict[str, float]:
        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        if pred_img.size != gt_img.size:
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        pred = self.to_tensor(pred_img).unsqueeze(0).to(self.device)
        gt = self.to_tensor(gt_img).unsqueeze(0).to(self.device)

        return {
            'PSNR': float(self.psnr(pred, gt).item()),
            'SSIM': float(self.ssim(pred, gt).item()),
            'VIF': float(self.vif(pred, gt).item()),
            'FSIM': float(self.fsim(pred, gt).item()),
            'NIQE': float(self.niqe(pred).item()),
        }


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'mean': 0.0, 'std': 0.0}
    arr = torch.tensor(values, dtype=torch.float32)
    return {'mean': float(arr.mean().item()), 'std': float(arr.std(unbiased=False).item())}


def resolve_gt(dataset_dir: Path, dataset_name: str, sample_stem: str) -> Path | None:
    if dataset_name == 'Rain100H':
        idx = sample_stem.split('-')[-1]
        candidates = list(dataset_dir.glob(f'norain-{idx}.*'))
        return candidates[0] if candidates else None

    if dataset_name == 'rain100H_train':
        gt_dir = dataset_dir / 'norain'
        candidates = list(gt_dir.glob(f'{sample_stem}.*'))
        return candidates[0] if candidates else None

    return None


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = (project_root / args.dataset_root).resolve()
    bench_root = (project_root / args.benchmark_root).resolve()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suite = MetricSuite(device)

    overall = {}

    for ds in args.datasets:
        ds_dataset_dir = dataset_root / ds
        ds_bench_dir = bench_root / ds
        restored_root = ds_bench_dir / 'restored'
        if not restored_root.exists():
            print(f'[WARN] restored folder missing: {restored_root}')
            continue

        rows = []
        sample_dirs = sorted([p for p in restored_root.iterdir() if p.is_dir()])
        for sample_dir in sample_dirs:
            sample_stem = sample_dir.name
            pred = sample_dir / 'final_output.png'
            if not pred.exists():
                continue

            gt = resolve_gt(ds_dataset_dir, ds, sample_stem)
            if gt is None or not gt.exists():
                continue

            m = suite.evaluate(pred, gt)
            rows.append({
                'dataset': ds,
                'sample': sample_stem,
                'gt_path': str(gt),
                'output_path': str(pred),
                **m,
            })

        out_csv = ds_bench_dir / f'per_image_metrics_{args.out_suffix}.csv'
        out_json = ds_bench_dir / f'summary_{args.out_suffix}.json'

        with out_csv.open('w', newline='', encoding='utf-8') as f:
            fieldnames = ['dataset', 'sample', 'gt_path', 'output_path', 'PSNR', 'SSIM', 'VIF', 'FSIM', 'NIQE']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        summary = {
            'dataset': ds,
            'num_samples': len(rows),
            'metrics': {
                'PSNR': mean_std([r['PSNR'] for r in rows]),
                'SSIM': mean_std([r['SSIM'] for r in rows]),
                'VIF': mean_std([r['VIF'] for r in rows]),
                'FSIM': mean_std([r['FSIM'] for r in rows]),
                'NIQE': mean_std([r['NIQE'] for r in rows]),
            }
        }
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        overall[ds] = summary
        print(f"[PARTIAL] {ds}: n={len(rows)}, PSNR={summary['metrics']['PSNR']['mean']:.3f}, SSIM={summary['metrics']['SSIM']['mean']:.4f}, NIQE={summary['metrics']['NIQE']['mean']:.3f}")

    overall_path = bench_root / f'overall_summary_{args.out_suffix}.json'
    overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Overall partial summary: {overall_path}')


if __name__ == '__main__':
    main()

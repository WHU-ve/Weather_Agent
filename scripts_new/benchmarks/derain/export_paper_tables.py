#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Export paper-style benchmark tables (Markdown + LaTeX).')
    parser.add_argument('--summary_json', required=True, help='Path to overall_summary.json')
    parser.add_argument('--method_name', default='WeatherAgent (Ours)', help='Method name in the table')
    parser.add_argument('--output_dir', default='', help='Output directory (default: same as summary_json)')
    parser.add_argument('--float_digits', type=int, default=4, help='Decimal places for table values')
    return parser.parse_args()


def fmt(x, d):
    return f'{x:.{d}f}'


def main():
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary json not found: {summary_path}')

    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text(encoding='utf-8'))
    if not data:
        raise ValueError('Empty summary json.')

    datasets = sorted(data.keys())
    metrics = ['PSNR', 'SSIM', 'VIF', 'FSIM', 'NIQE']

    rows = []
    for ds in datasets:
        item = data[ds]
        m = item.get('metrics', {})
        row = {
            'dataset': ds,
            'PSNR': float(m.get('PSNR', {}).get('mean', 0.0)),
            'SSIM': float(m.get('SSIM', {}).get('mean', 0.0)),
            'VIF': float(m.get('VIF', {}).get('mean', 0.0)),
            'FSIM': float(m.get('FSIM', {}).get('mean', 0.0)),
            'NIQE': float(m.get('NIQE', {}).get('mean', 0.0)),
        }
        rows.append(row)

    avg_row = {'dataset': 'Mean(2 datasets)'}
    for k in metrics:
        avg_row[k] = sum(r[k] for r in rows) / max(len(rows), 1)

    md_lines = []
    md_lines.append('| Method | Dataset | PSNR↑ | SSIM↑ | VIF↑ | FSIM↑ | NIQE↓ |')
    md_lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        md_lines.append(
            f"| {args.method_name} | {r['dataset']} | {fmt(r['PSNR'], args.float_digits)} | "
            f"{fmt(r['SSIM'], args.float_digits)} | {fmt(r['VIF'], args.float_digits)} | "
            f"{fmt(r['FSIM'], args.float_digits)} | {fmt(r['NIQE'], args.float_digits)} |"
        )
    md_lines.append(
        f"| {args.method_name} | **{avg_row['dataset']}** | **{fmt(avg_row['PSNR'], args.float_digits)}** | "
        f"**{fmt(avg_row['SSIM'], args.float_digits)}** | **{fmt(avg_row['VIF'], args.float_digits)}** | "
        f"**{fmt(avg_row['FSIM'], args.float_digits)}** | **{fmt(avg_row['NIQE'], args.float_digits)}** |"
    )

    latex_lines = []
    latex_lines.append('\\begin{table}[t]')
    latex_lines.append('\\centering')
    latex_lines.append('\\caption{Deraining results on two datasets.}')
    latex_lines.append('\\label{tab:derain_two_datasets}')
    latex_lines.append('\\begin{tabular}{l l c c c c c}')
    latex_lines.append('\\hline')
    latex_lines.append('Method & Dataset & PSNR$\\uparrow$ & SSIM$\\uparrow$ & VIF$\\uparrow$ & FSIM$\\uparrow$ & NIQE$\\downarrow$ \\\\')
    latex_lines.append('\\hline')
    for r in rows:
        latex_lines.append(
            f"{args.method_name} & {r['dataset']} & {fmt(r['PSNR'], args.float_digits)} & "
            f"{fmt(r['SSIM'], args.float_digits)} & {fmt(r['VIF'], args.float_digits)} & "
            f"{fmt(r['FSIM'], args.float_digits)} & {fmt(r['NIQE'], args.float_digits)} \\\\"
        )
    latex_lines.append('\\hline')
    latex_lines.append(
        f"{args.method_name} & Mean(2 datasets) & "
        f"\\textbf{{{fmt(avg_row['PSNR'], args.float_digits)}}} & "
        f"\\textbf{{{fmt(avg_row['SSIM'], args.float_digits)}}} & "
        f"\\textbf{{{fmt(avg_row['VIF'], args.float_digits)}}} & "
        f"\\textbf{{{fmt(avg_row['FSIM'], args.float_digits)}}} & "
        f"\\textbf{{{fmt(avg_row['NIQE'], args.float_digits)}}} \\\\"
    )
    latex_lines.append('\\hline')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')

    md_path = output_dir / 'paper_table_derain_two_datasets.md'
    tex_path = output_dir / 'paper_table_derain_two_datasets.tex'
    json_path = output_dir / 'paper_table_derain_two_datasets.json'

    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    tex_path.write_text('\n'.join(latex_lines) + '\n', encoding='utf-8')
    json_path.write_text(json.dumps({'rows': rows, 'mean': avg_row}, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Markdown table: {md_path}')
    print(f'LaTeX table: {tex_path}')
    print(f'JSON table data: {json_path}')


if __name__ == '__main__':
    main()

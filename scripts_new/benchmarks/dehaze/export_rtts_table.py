#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Export RTTS dehaze no-reference table (Markdown + LaTeX).')
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--method_name', default='WeatherAgent (Ours)')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--float_digits', type=int, default=4)
    return parser.parse_args()


def fmt(x, d):
    return f'{x:.{d}f}'


def main():
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    data = json.loads(summary_path.read_text(encoding='utf-8'))

    out_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    m = data['metrics']
    row = {
        'Dataset': data.get('dataset', 'RTTS'),
        'n': int(data.get('num_samples', 0)),
        'HazeBefore': float(m['HazeProbBefore']['mean']),
        'HazeAfter': float(m['HazeProbAfter']['mean']),
        'DeltaHaze': float(m['DeltaHazeProb']['mean']),
        'NIQE': float(m['NIQE']['mean']),
        'MUSIQ': float(m['MUSIQ']['mean']),
        'CLIPIQA': float(m['CLIPIQA']['mean']),
    }

    md = [
        '| Method | Dataset | n | HazeProb(before)↓ | HazeProb(after)↓ | ΔHazeProb↑ | NIQE↓ | MUSIQ↑ | CLIPIQA↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| {args.method_name} | {row['Dataset']} | {row['n']} | {fmt(row['HazeBefore'], args.float_digits)} | "
        f"{fmt(row['HazeAfter'], args.float_digits)} | {fmt(row['DeltaHaze'], args.float_digits)} | "
        f"{fmt(row['NIQE'], args.float_digits)} | {fmt(row['MUSIQ'], args.float_digits)} | {fmt(row['CLIPIQA'], args.float_digits)} |",
    ]

    tex = [
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{No-reference dehazing results on RTTS.}',
        '\\label{tab:rtts_dehaze}',
        '\\begin{tabular}{l l c c c c c c c}',
        '\\hline',
        'Method & Dataset & n & HazeProb(bef.)$\\downarrow$ & HazeProb(aft.)$\\downarrow$ & $\\Delta$HazeProb$\\uparrow$ & NIQE$\\downarrow$ & MUSIQ$\\uparrow$ & CLIPIQA$\\uparrow$ \\\\',
        '\\hline',
        f"{args.method_name} & {row['Dataset']} & {row['n']} & {fmt(row['HazeBefore'], args.float_digits)} & "
        f"{fmt(row['HazeAfter'], args.float_digits)} & {fmt(row['DeltaHaze'], args.float_digits)} & "
        f"{fmt(row['NIQE'], args.float_digits)} & {fmt(row['MUSIQ'], args.float_digits)} & {fmt(row['CLIPIQA'], args.float_digits)} \\\\ ",
        '\\hline',
        '\\end{tabular}',
        '\\end{table}',
    ]

    (out_dir / 'paper_table_rtts_dehaze.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out_dir / 'paper_table_rtts_dehaze.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')

    print(f"Markdown table: {out_dir / 'paper_table_rtts_dehaze.md'}")
    print(f"LaTeX table: {out_dir / 'paper_table_rtts_dehaze.tex'}")


if __name__ == '__main__':
    main()

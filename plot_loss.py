#!/usr/bin/env python3
"""Plot the training loss curve from checkpoints/loss_history.jsonl.

train.py appends one {"kind": "train", ...} row per progress window (~2 s)
and {"kind": "eval", ...} rows at checkpoint cadence, both keyed by the
cumulative number of samples trained.  This script renders the loss against
that count: faint raw train rows, a thicker rolling-mean train line and a
separate eval series.

Usage:
    python plot_loss.py
    python plot_loss.py --smooth 20 --out curve.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HISTORY_PATH = os.path.join("checkpoints", "loss_history.jsonl")


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot the training loss curve")
    parser.add_argument(
        "--history",
        default=HISTORY_PATH,
        help="loss history written by train.py (default: checkpoints/loss_history.jsonl)",
    )
    parser.add_argument("--out", default="loss_curve.png", help="output image (default: loss_curve.png)")
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=10,
        help="rolling window over train rows (default: 10; 1 disables smoothing)",
    )
    return parser.parse_args(argv)


def read_history(path):
    train_rows, eval_rows = [], []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            loss = row.get("loss")
            if loss is None:
                continue
            series = eval_rows if row.get("kind") == "eval" else train_rows
            series.append((row["samples"], loss))
    return train_rows, eval_rows


def rolling_mean(values, window):
    return [sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def plot_rows(axis, rows, **kwargs):
    axis.plot([x for x, _ in rows], [y for _, y in rows], **kwargs)


def main():
    args = parse_args()
    if not os.path.isfile(args.history):
        raise SystemExit(f"error: loss history not found at {args.history}; run train.py first")
    train_rows, eval_rows = read_history(args.history)
    if not train_rows and not eval_rows:
        raise SystemExit(f"error: {args.history} contains no loss rows yet")

    fig, axis = plt.subplots(figsize=(10, 6), dpi=150)
    if len(train_rows) >= 2:
        plot_rows(axis, train_rows, color="#94a3b8", linewidth=0.8, alpha=0.6, label="train")
        if args.smooth > 1:
            smooth = list(zip((x for x, _ in train_rows), rolling_mean([y for _, y in train_rows], args.smooth)))
            plot_rows(axis, smooth, color="#2563eb", linewidth=1.8, label=f"train ({args.smooth}-row mean)")
    if eval_rows:
        plot_rows(
            axis,
            eval_rows,
            color="#dc2626",
            linewidth=1.2,
            marker="o",
            markersize=3,
            label="eval (checkpoints)",
        )
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1e6:g}M"))
    axis.set_xlabel("samples trained")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out} ({len(train_rows)} train rows, {len(eval_rows)} eval rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure move-match quality and Elo fidelity of a trained checkpoint.

Three reports on data/eval.bin:

1. Move-match top-1 / top-3 and cross entropy against the human move,
   bucketed by the Elo of the player to move.  Humans agree with other
   humans on roughly 40-55% of moves (top-1), which is the reference band.
2. Cross-Elo CE matrix: the same positions re-evaluated with the Elo
   token overwritten.  A visible diagonal means the model uses Elo.
3. Elo KL probe: mean KL(p_low || p_high) over legal moves on the same
   positions.  Values near zero mean the model ignores Elo.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/ckpt_0000000100000.pt --samples 50000
"""

import argparse
import sys

import numpy as np
import torch

from model import ChessTransformer, ModelConfig, make_batch
from train import (
    CHECKPOINT_DIR,
    DATA_DIR,
    latest_checkpoint_path,
    load_split,
    legal_lists,
    resolve_device,
)

ELO_GRID = (8, 11, 14, 17, 20)  # 800, 1100, 1400, 1700, 2000
KL_PAIRS = ((8, 20), (11, 17), (14, 20))
BUCKET_WIDTH = 2  # 200 Elo points


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate move-match and Elo fidelity")
    parser.add_argument("--checkpoint", help="checkpoint file (default: latest in checkpoints/)")
    parser.add_argument("--device", default="gpu:0", help="cpu or gpu:N (default: gpu:0)")
    parser.add_argument("--samples", type=int, default=20_000, help="eval samples for move-match (default: 20000)")
    parser.add_argument("--cross-elo-samples", dest="cross_elo_samples", type=int, default=5_000)
    parser.add_argument("--kl-samples", dest="kl_samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ChessTransformer(ModelConfig.from_dict(payload["state"]["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def run_logits(model, batch, device, autocast_enabled):
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            return model(batch).float()


def sample_stats(logits_row, ids, target):
    """(ce, rank) of one sample against its legal move id list."""
    ids = [int(i) for i in ids]
    values = logits_row[ids]
    t_index = ids.index(int(target))
    ce = float(torch.logsumexp(values, dim=0) - values[t_index])
    rank = int((torch.argsort(values, descending=True) == t_index).nonzero()[0, 0])
    return ce, rank


def move_match_report(model, records, legal_ids, offsets, device, autocast_enabled, batch_size, limit):
    count = min(limit, len(records))
    print(f"\nmove-match on {count:,} samples (conditioned on the player's Elo)")
    print(f"{'bucket':>10} {'n':>8} {'top1':>7} {'top3':>7} {'ce':>7}")
    buckets = {}
    total_ce = top1 = top3 = 0
    for start in range(0, count, batch_size):
        part = np.arange(start, min(start + batch_size, count))
        batch = make_batch(records[part], device)
        logits = run_logits(model, batch, device, autocast_enabled)
        legal = legal_lists(legal_ids, offsets, part)
        for row, index in enumerate(part):
            ce, rank = sample_stats(logits[row], legal[row], records["target"][index])
            key = int(records["elo"][index]) // BUCKET_WIDTH
            slot = buckets.setdefault(key, [0.0, 0, 0, 0])
            slot[0] += ce
            slot[1] += int(rank == 0)
            slot[2] += int(rank < 3)
            slot[3] += 1
            total_ce += ce
            top1 += int(rank == 0)
            top3 += int(rank < 3)
    for key in sorted(buckets):
        ce_sum, hit1, hit3, n = buckets[key]
        print(f"{key * BUCKET_WIDTH * 100:>10} {n:>8,} {hit1 / n:>7.1%} {hit3 / n:>7.1%} {ce_sum / n:>7.3f}")
    print(f"{'overall':>10} {count:>8,} {top1 / count:>7.1%} {top3 / count:>7.1%} {total_ce / count:>7.3f}")


def cross_elo_report(model, records, legal_ids, offsets, device, autocast_enabled, batch_size, limit):
    count = min(limit, len(records))
    legal = legal_lists(legal_ids, offsets, np.arange(count))
    sample_buckets = [int(elo) // BUCKET_WIDTH for elo in records["elo"][:count]]
    columns = sorted(set(sample_buckets))
    matrix = {(g, b): [0.0, 0] for g in ELO_GRID for b in columns}

    print(f"\ncross-Elo CE matrix on {count:,} samples (rows: Elo token, cols: sample's own bucket)")
    for g in ELO_GRID:
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = make_batch(records[start:end], device)
            batch["elo"] = torch.full_like(batch["elo"], g)
            logits = run_logits(model, batch, device, autocast_enabled)
            for row in range(end - start):
                index = start + row
                ce, _ = sample_stats(logits[row], legal[index], records["target"][index])
                slot = matrix[(g, sample_buckets[index])]
                slot[0] += ce
                slot[1] += 1

    print(" " * 8 + "".join(f"{b * BUCKET_WIDTH * 100:>8}" for b in columns))
    for g in ELO_GRID:
        cells = []
        for b in columns:
            ce_sum, n = matrix[(g, b)]
            cells.append(f"{ce_sum / n:>8.3f}" if n else f"{'-':>8}")
        print(f"{g * 100:>8}" + "".join(cells))


def kl_probe(model, records, legal_ids, offsets, device, autocast_enabled, batch_size, limit):
    count = min(limit, len(records))
    legal = legal_lists(legal_ids, offsets, np.arange(count))
    elos = sorted({e for pair in KL_PAIRS for e in pair})
    sums = {pair: 0.0 for pair in KL_PAIRS}

    print(f"\nElo KL probe on {count:,} samples (mean KL over legal moves)")
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        chunk = records[start:end]
        probs = {}
        for g in elos:
            batch = make_batch(chunk, device)
            batch["elo"] = torch.full_like(batch["elo"], g)
            logits = run_logits(model, batch, device, autocast_enabled)
            probs[g] = [
                torch.softmax(logits[row, [int(i) for i in legal[start + row]]], dim=0)
                for row in range(end - start)
            ]
        for pair in KL_PAIRS:
            for row in range(end - start):
                p = probs[pair[0]][row]
                q = probs[pair[1]][row]
                sums[pair] += float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())
    for pair in KL_PAIRS:
        print(f"  KL(elo={pair[0] * 100} || elo={pair[1] * 100}) = {sums[pair] / count:.4f} nats")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    autocast_enabled = device.type == "cuda"

    path = args.checkpoint or latest_checkpoint_path()
    if path is None:
        sys.exit(f"error: no checkpoint found in {CHECKPOINT_DIR}/")
    print(f"checkpoint {path}")
    model = load_model(path, device)

    meta, records, legal_ids, offsets = load_split(DATA_DIR, "eval")
    print(f"eval samples available: {len(records):,}")

    move_match_report(model, records, legal_ids, offsets, device, autocast_enabled, args.batch_size, args.samples)
    cross_elo_report(
        model, records, legal_ids, offsets, device, autocast_enabled, args.batch_size, args.cross_elo_samples
    )
    kl_probe(model, records, legal_ids, offsets, device, autocast_enabled, args.batch_size, args.kl_samples)


if __name__ == "__main__":
    main()

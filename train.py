#!/usr/bin/env python3
"""Train the chess move model on per-move binary train/eval shards.

Reads data/train.bin (see fen_codec.py) together with its precomputed
legal-move CSR arrays, shuffles the sample order every epoch and trains
with AdamW: constant weight decay on weights and embeddings, linear
learning-rate warmup followed by cosine decay, gradient clipping at 1
and bf16 autocast on CUDA.  Every --checkpoint-cadence samples the model
is evaluated on a fixed slice of data/eval.bin and a checkpoint is
written to checkpoints/ containing the optimizer state and the exact
epoch/shuffle position so training can be resumed with --resume.  If the
training data changed since the checkpoint (different meta.json
fingerprint) the weights are kept but the stream position is reset.

Usage:
    python train.py --epochs 1 --lr 3e-4 --device gpu:0
    python train.py --resume --checkpoint-cadence 1000000
"""

import argparse
import glob
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch

from fen_codec import RECORD_DTYPE
from model import ChessTransformer, ModelConfig, legal_move_loss, make_batch

DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
MOVES_PATH = "moves.json"
DEFAULT_LR = 3e-4
DEFAULT_WD = 0.1
GRAD_CLIP = 1.0
BATCH_SIZE = 512
WARMUP_STEPS = 2000
MIN_LR_RATIO = 0.1
EVAL_SAMPLES = 20_000
KEEP_CHECKPOINTS = 2
PROGRESS_INTERVAL = 2.0


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the chess move model")
    parser.add_argument("--epochs", type=positive_int, default=1, help="passes over the training data (default: 1)")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="peak learning rate (default: 3e-4, or the checkpoint's lr when resuming)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=None,
        help="AdamW weight decay on weights and embeddings (default: 0.1, or the checkpoint's wd when resuming)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument(
        "--max-samples",
        type=positive_int,
        default=None,
        help="cap the number of training samples per epoch (default: all)",
    )
    parser.add_argument("--device", default="gpu:0", help="cpu or gpu:N (default: gpu:0)")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in checkpoints/")
    parser.add_argument(
        "--checkpoint-cadence",
        dest="checkpoint_cadence",
        type=positive_int,
        default=100_000,
        help="create a checkpoint every N training samples (default: 100000)",
    )
    return parser.parse_args(argv)


def resolve_device(value):
    value = value.strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "gpu":
        value = "gpu:0"
    if value.startswith("gpu:"):
        try:
            index = int(value[4:])
        except ValueError:
            index = -1
        if index < 0:
            raise argparse.ArgumentTypeError(f"invalid device: {value}")
        if not torch.cuda.is_available():
            sys.exit("error: CUDA is not available; use --device cpu")
        if index >= torch.cuda.device_count():
            sys.exit(f"error: no CUDA device with index {index}")
        return torch.device(f"cuda:{index}")
    raise argparse.ArgumentTypeError(f"invalid device: {value}")


def load_split(data_dir, stem):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
    records = np.fromfile(os.path.join(data_dir, f"{stem}.bin"), dtype=RECORD_DTYPE)
    legal_ids = np.fromfile(os.path.join(data_dir, f"{stem}.legal.bin"), dtype=np.uint16)
    offsets = np.fromfile(os.path.join(data_dir, f"{stem}.legal_off.bin"), dtype=np.uint32)
    if len(records) != meta[stem]["samples"] or len(offsets) != len(records) + 1:
        sys.exit(f"error: {stem} shards do not match data/meta.json")
    return meta, records, legal_ids, offsets


def legal_lists(legal_ids, offsets, indices):
    return [legal_ids[offsets[i] : offsets[i + 1]] for i in indices]


def lr_at(step, total_steps, peak):
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS))
    floor = peak * MIN_LR_RATIO
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model, lr, wd):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def checkpoint_path(global_samples):
    return os.path.join(CHECKPOINT_DIR, f"ckpt_{global_samples:012d}.pt")


def save_checkpoint(model, optimizer, state):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(state["global_samples"])
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        },
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def prune_checkpoints():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt")))
    for path in checkpoints[:-KEEP_CHECKPOINTS]:
        os.remove(path)


def clear_checkpoints():
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt")):
        os.remove(path)


def latest_checkpoint_path():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt")))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path, fingerprint, config, model, optimizer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    if state["config"] != config.as_dict():
        sys.exit("error: model architecture changed since the checkpoint was written")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"]["torch"])
    if payload["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    random.setstate(payload["rng"]["python"])
    if state["fingerprint"] == fingerprint:
        print(
            f"resuming {path} at epoch {state['epoch']}, "
            f"sample {state['global_samples']:,}"
        )
        return state
    print(
        f"warning: training data changed since {path}; keeping weights, "
        "restarting the sample stream from the beginning"
    )
    return None


def make_state(fingerprint, config, epoch, step_in_epoch, perm_seed, eval_loss, lr, wd, global_steps, global_samples):
    return {
        "fingerprint": fingerprint,
        "config": config.as_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "perm_seed": perm_seed,
        "eval_loss": eval_loss,
        "lr": lr,
        "wd": wd,
        "global_steps": global_steps,
        "global_samples": global_samples,
    }


def evaluate_loss(model, device, records, legal_ids, offsets, autocast_enabled):
    count = min(EVAL_SAMPLES, len(records))
    indices = np.arange(count)
    total = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, count, BATCH_SIZE):
            part = indices[start : start + BATCH_SIZE]
            batch = make_batch(records[part], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(batch)
                loss = legal_move_loss(logits, legal_lists(legal_ids, offsets, part), batch["targets"])
            total += loss.item() * len(part)
            seen += len(part)
    model.train()
    return total / seen if seen else None


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def handle_sigterm(_signum, _frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, handle_sigterm)
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)

    train_meta, train_records, train_legal, train_offsets = load_split(DATA_DIR, "train")
    eval_meta, eval_records, eval_legal, eval_offsets = (None, None, None, None)
    if os.path.isfile(os.path.join(DATA_DIR, "eval.bin")):
        eval_meta, eval_records, eval_legal, eval_offsets = load_split(DATA_DIR, "eval")

    n = len(train_records)
    if args.max_samples is not None:
        n = min(n, args.max_samples)
    fingerprint = train_meta["fingerprint"]
    print(f"train samples: {len(train_records):,} (using {n:,})  eval samples: {len(eval_records):,}")

    with open(MOVES_PATH, "r", encoding="utf-8") as file:
        moves_fingerprint = json.load(file)["fingerprint"]
    config = ModelConfig()
    model = ChessTransformer(config).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

    lr = args.lr if args.lr is not None else DEFAULT_LR
    wd = args.wd if args.wd is not None else DEFAULT_WD
    optimizer = build_optimizer(model, lr, wd)

    resume_state = None
    if args.resume:
        path = latest_checkpoint_path()
        if path is None:
            sys.exit("error: --resume but checkpoints/ contains no checkpoint")
        resume_state = load_checkpoint(path, fingerprint, config, model, optimizer)
        if resume_state is not None:
            if resume_state.get("moves_fingerprint") not in (None, moves_fingerprint):
                sys.exit("error: moves.json changed since the checkpoint was written")
            if args.lr is not None:
                lr = args.lr
            if args.wd is not None:
                wd = args.wd
            for index, group in enumerate(optimizer.param_groups):
                group["lr"] = lr
                group["weight_decay"] = wd if index == 0 else 0.0
            if resume_state["epoch"] >= args.epochs:
                print("training already complete")
                return
    if resume_state is None:
        # Fresh stream: drop checkpoints from earlier runs so prune_checkpoints
        # cannot delete new ones just because their sample counts are smaller.
        clear_checkpoints()

    print(f"optimizer peak learning rate {lr:.4g}, weight decay {wd:.4g}")

    batch_size = args.batch_size
    steps_per_epoch = max(1, n // batch_size)
    total_steps = steps_per_epoch * args.epochs
    start_epoch = resume_state["epoch"] if resume_state else 0
    start_step = resume_state["step_in_epoch"] if resume_state else 0
    global_steps = resume_state["global_steps"] if resume_state else 0
    global_samples = resume_state["global_samples"] if resume_state else 0
    done_samples = global_samples
    last_saved_global = None
    next_trigger = ((global_samples // args.checkpoint_cadence) + 1) * args.checkpoint_cadence

    autocast_enabled = device.type == "cuda"
    started_at = time.monotonic()
    last_progress = started_at
    loss_sum = 0.0
    loss_count = 0
    epoch = start_epoch
    step_in_epoch = start_step
    perm_seed = resume_state["perm_seed"] if resume_state else None

    try:
        for epoch in range(start_epoch, args.epochs):
            if resume_state is not None and epoch == start_epoch and perm_seed is not None:
                seed = perm_seed
            else:
                seed = random.randrange(2**32)
            perm_seed = seed
            order = np.random.default_rng(seed).permutation(len(train_records))[:n]
            start_step = resume_state["step_in_epoch"] if (resume_state and epoch == start_epoch) else 0
            step_in_epoch = start_step

            for step_in_epoch in range(start_step, steps_per_epoch):
                part = order[step_in_epoch * batch_size : (step_in_epoch + 1) * batch_size]
                batch = make_batch(train_records[part], device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    logits = model(batch)
                    loss = legal_move_loss(logits, legal_lists(train_legal, train_offsets, part), batch["targets"])
                loss.backward()
                loss_sum += loss.item() * len(part)
                loss_count += len(part)

                current_lr = lr_at(global_steps, total_steps, lr)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_steps += 1
                global_samples += len(part)

                now = time.monotonic()
                if now - last_progress >= PROGRESS_INTERVAL:
                    rate = (global_samples - done_samples) / max(now - started_at, 1e-9)
                    remaining = max(total_steps * batch_size - global_samples, 0)
                    mean_loss = loss_sum / max(loss_count, 1)
                    print(
                        f"epoch {epoch + 1}/{args.epochs}  "
                        f"samples {global_samples:,}  "
                        f"loss {mean_loss:.4f}  lr {current_lr:.2e}  "
                        f"{rate:,.0f} samples/s  "
                        f"ETA {format_duration(remaining / rate) if rate > 0 else 0}"
                    )
                    last_progress = now
                    loss_sum = 0.0
                    loss_count = 0

                if global_samples >= next_trigger:
                    while global_samples >= next_trigger:
                        next_trigger += args.checkpoint_cadence
                    eval_loss = None
                    if eval_records is not None:
                        eval_loss = evaluate_loss(
                            model, device, eval_records, eval_legal, eval_offsets, autocast_enabled
                        )
                    next_epoch = epoch
                    next_step = step_in_epoch + 1
                    if next_step >= steps_per_epoch:
                        next_epoch, next_step = epoch + 1, 0
                    state = make_state(
                        fingerprint, config, next_epoch, next_step, perm_seed,
                        eval_loss, lr, wd, global_steps, global_samples,
                    )
                    state["moves_fingerprint"] = moves_fingerprint
                    path = save_checkpoint(model, optimizer, state)
                    prune_checkpoints()
                    last_saved_global = global_samples
                    print(f"checkpoint {path}  eval loss {eval_loss}")
            resume_state = None
    except KeyboardInterrupt:
        print("\ninterrupted; saving checkpoint")
        state = make_state(
            fingerprint, config, epoch, step_in_epoch, perm_seed,
            None, lr, wd, global_steps, global_samples,
        )
        state["moves_fingerprint"] = moves_fingerprint
        if last_saved_global != global_samples:
            save_checkpoint(model, optimizer, state)
            prune_checkpoints()
        raise SystemExit(130)

    if last_saved_global != global_samples:
        eval_loss = None
        if eval_records is not None:
            eval_loss = evaluate_loss(
                model, device, eval_records, eval_legal, eval_offsets, autocast_enabled
            )
        state = make_state(
            fingerprint, config, args.epochs, 0, perm_seed,
            eval_loss, lr, wd, global_steps, global_samples,
        )
        state["moves_fingerprint"] = moves_fingerprint
        path = save_checkpoint(model, optimizer, state)
        prune_checkpoints()
        print(f"\ncheckpoint {path}  eval loss {eval_loss}")
    print("training complete")


if __name__ == "__main__":
    main()

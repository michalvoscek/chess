#!/usr/bin/env python3
"""Train the chess move model on per-move train/eval jsonl data.

Streams data/train.jsonl sequentially (no shuffling) in micro-batches of 256
with gradient accumulation 2 for an effective batch of 512, using AdamW with
weight decay 0.1 on weights and embeddings only, a constant learning rate and
gradient clipping at 1.  Every --checkpoint-cadence samples the model is
evaluated on the next proportional slice of data/eval.jsonl (cross entropy
over legal moves) and a checkpoint is written to checkpoints/ containing the
optimizer state and the exact stream position, so training can be resumed
with --resume.  If the training data changed since the checkpoint (different
sha256 fingerprint) the weights are kept but the stream position is reset.

Usage:
    python train.py --epochs 1 --lr 3e-4 --device gpu:0
    python train.py --resume --checkpoint-cadence 1000000
"""

import argparse
from collections import deque
import glob
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time

import torch

from model import (
    ChessTransformer,
    ModelConfig,
    legal_move_loss,
    load_vocab,
    tokenize_sample,
)

VOCAB_PATH = "vocab.json"
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
MICRO_BATCH = 256
ACCUM_STEPS = 2
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHUNK_LINES = 2048
WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
TASK_WINDOW = WORKERS * 2
KEEP_CHECKPOINTS = 2
SCAN_CHUNK = 1 << 22
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
    parser.add_argument("--lr", type=float, default=3e-4, help="constant learning rate (default: 3e-4)")
    parser.add_argument("--device", default="gpu:0", help="cpu or gpu:N (default: gpu:0)")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in checkpoints/")
    parser.add_argument(
        "--checkpoint-cadence",
        dest="checkpoint_cadence",
        type=positive_int,
        help="create a checkpoint every N training samples (default: once per epoch)",
    )
    args = parser.parse_args(argv)
    if args.checkpoint_cadence and args.checkpoint_cadence < MICRO_BATCH * ACCUM_STEPS:
        parser.error(f"--checkpoint-cadence must be at least {MICRO_BATCH * ACCUM_STEPS}")
    return args


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


def scan_file(path):
    """Single pass over a file returning (sha256 hex digest, line count)."""
    digest = hashlib.sha256()
    lines = 0
    with open(path, "rb") as file:
        while chunk := file.read(SCAN_CHUNK):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def iter_chunks(path, start_line, chunk_size):
    """Yield (first_line_number, lines) chunks starting at start_line."""
    with open(path, "r", encoding="utf-8") as file:
        if start_line:
            deque(itertools.islice(file, start_line), maxlen=0)
        chunk_start = start_line
        buffer = []
        for line in file:
            buffer.append(line)
            if len(buffer) == chunk_size:
                yield chunk_start, buffer
                chunk_start += len(buffer)
                buffer = []
        if buffer:
            yield chunk_start, buffer


def imap_window(pool, func, iterable, window):
    """imap with bounded prefetch: at most `window` tasks in flight."""
    iterator = iter(iterable)
    pending = deque()
    while True:
        while len(pending) < window:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(pool.apply_async(func, (item,)))
        if not pending:
            return
        yield pending.popleft().get()


_worker_token_to_id = None


def _init_worker():
    global _worker_token_to_id
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_token_to_id = load_vocab(VOCAB_PATH)["token_to_id"]


def _worker_legal_ids(sample):
    from fen_codec import sample_to_board
    from model import legal_move_ids

    return legal_move_ids(sample, _worker_token_to_id)


def _prepare_chunk(chunk):
    """Tokenize one chunk of raw jsonl lines into trainable samples."""
    chunk_start, lines = chunk
    prepared = []
    for offset, line in enumerate(lines):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        tokenized = tokenize_sample(sample, _worker_token_to_id)
        if tokenized is None:
            continue
        ids, target = tokenized
        legal = _worker_legal_ids(sample)
        if target not in legal:
            continue
        prepared.append((chunk_start + offset, ids, target, legal))
    return prepared


def build_optimizer(model, lr):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def make_batch(samples, device):
    tokens = torch.tensor([sample[1] for sample in samples], dtype=torch.long, device=device)
    targets = torch.tensor([sample[2] for sample in samples], dtype=torch.long, device=device)
    legal = [sample[3] for sample in samples]
    return tokens, targets, legal


def run_micro(model, batch, device, autocast_enabled):
    tokens, targets, legal = batch
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        logits = model(tokens)
        loss = legal_move_loss(logits, legal, targets)
    return loss


def optimizer_step(model, optimizer):
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def evaluate_slice(model, device, pool, eval_path, eval_offset, slice_size):
    """Mean loss over the next slice_size eval samples starting at eval_offset.

    Returns (mean_loss, new_offset).  Wraps around once at end of file.
    """
    model.eval()
    total_loss = 0.0
    counted = 0
    position = eval_offset

    def chunk_stream():
        remaining = slice_size
        pos = eval_offset
        wrapped = False
        while remaining > 0:
            yielded = False
            for chunk_start, lines in iter_chunks(eval_path, pos, CHUNK_LINES):
                take = lines[:remaining]
                yield chunk_start, take
                pos = chunk_start + len(take)
                remaining -= len(take)
                yielded = True
                if remaining <= 0:
                    return
            if not yielded or wrapped:
                return
            wrapped = True
            pos = 0

    batches = []
    with torch.no_grad():
        for prepared in imap_window(pool, _prepare_chunk, chunk_stream(), TASK_WINDOW):
            batches.extend(prepared)
            position = max(position, prepared[-1][0] + 1)
            while len(batches) >= MICRO_BATCH:
                micro, batches = batches[:MICRO_BATCH], batches[MICRO_BATCH:]
                loss = run_micro(model, make_batch(micro, device), device, device.type == "cuda")
                total_loss += loss.item() * len(micro)
                counted += len(micro)
        if batches:
            loss = run_micro(model, make_batch(batches, device), device, device.type == "cuda")
            total_loss += loss.item() * len(batches)
            counted += len(batches)
    model.train()
    if counted == 0:
        return None, position
    return total_loss / counted, position


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


def latest_checkpoint_path():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt")))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path, fingerprint, vocab_fingerprint, config, model, optimizer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    if state["vocab_fingerprint"] != vocab_fingerprint:
        sys.exit("error: vocab.json changed since the checkpoint was written")
    if state["config"] != config.as_dict():
        sys.exit("error: model architecture changed since the checkpoint was written")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"]["torch"])
    if payload["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    random.setstate(payload["rng"]["python"])
    if state["fingerprint"] == fingerprint:
        print(f"resuming {path} at epoch {state['epoch']}, sample {state['offset']:,}")
        return state["epoch"], state["offset"], state["eval_offset"]
    print(
        f"warning: training data changed since {path}; keeping weights, "
        "restarting the sample stream from the beginning"
    )
    return 0, 0, 0


def make_state(fingerprint, vocab_fingerprint, config, epoch, offset, eval_offset, eval_loss, lr, global_samples):
    return {
        "fingerprint": fingerprint,
        "vocab_fingerprint": vocab_fingerprint,
        "config": config.as_dict(),
        "epoch": epoch,
        "offset": offset,
        "eval_offset": eval_offset,
        "eval_loss": eval_loss,
        "lr": lr,
        "global_samples": global_samples,
    }


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)

    vocab = load_vocab(VOCAB_PATH)
    vocab_fingerprint = vocab["fingerprint"]
    config = ModelConfig(vocab_size=vocab["total_tokens"])

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    eval_path = os.path.join(DATA_DIR, "eval.jsonl")
    if not os.path.isfile(train_path):
        sys.exit(f"error: training data not found: {train_path}")

    print(f"scanning {train_path}")
    fingerprint, train_lines = scan_file(train_path)
    eval_lines = 0
    if os.path.isfile(eval_path):
        _, eval_lines = scan_file(eval_path)
    print(f"train samples: {train_lines:,}  eval samples: {eval_lines:,}")

    pool = multiprocessing.Pool(WORKERS, initializer=_init_worker)

    model = ChessTransformer(config).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")
    optimizer = build_optimizer(model, args.lr)

    start_epoch, start_offset, eval_offset = 0, 0, 0
    if args.resume:
        path = latest_checkpoint_path()
        if path is None:
            sys.exit("error: --resume but checkpoints/ contains no checkpoint")
        start_epoch, start_offset, eval_offset = load_checkpoint(
            path, fingerprint, vocab_fingerprint, config, model, optimizer
        )
        if start_epoch >= args.epochs:
            print("training already complete")
            return

    cadence = args.checkpoint_cadence or train_lines
    expected_checkpoints = max(1, math.ceil(args.epochs * train_lines / cadence))
    eval_slice_size = max(1, math.ceil(eval_lines / expected_checkpoints)) if eval_lines else 0
    next_trigger = ((start_offset // cadence) + 1) * cadence

    autocast_enabled = device.type == "cuda"
    total_samples = args.epochs * train_lines
    done_samples = start_epoch * train_lines + start_offset
    started_at = time.monotonic()
    last_progress = started_at
    loss_sum = 0.0
    loss_count = 0
    last_saved_global = None
    epoch = start_epoch
    lines_done = start_offset
    model.train()

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_offset = start_offset if epoch == start_epoch else 0
            lines_done = epoch_start_offset
            chunk_source = iter_chunks(train_path, epoch_start_offset, CHUNK_LINES)
            buffer = []
            micros_since_step = 0

            for prepared in imap_window(pool, _prepare_chunk, chunk_source, TASK_WINDOW):
                buffer.extend(prepared)
                while len(buffer) >= MICRO_BATCH:
                    micro, buffer = buffer[:MICRO_BATCH], buffer[MICRO_BATCH:]
                    loss = run_micro(model, make_batch(micro, device), device, autocast_enabled)
                    (loss / ACCUM_STEPS).backward()
                    loss_sum += loss.item() * len(micro)
                    loss_count += len(micro)
                    lines_done = micro[-1][0] + 1
                    done_samples = epoch * train_lines + lines_done
                    micros_since_step += 1
                    if micros_since_step < ACCUM_STEPS:
                        continue
                    optimizer_step(model, optimizer)
                    micros_since_step = 0

                    now = time.monotonic()
                    if now - last_progress >= PROGRESS_INTERVAL:
                        consumed = max(done_samples - start_epoch * train_lines - start_offset, 1)
                        rate = consumed / (now - started_at)
                        remaining = max(total_samples - done_samples, 0)
                        mean_loss = loss_sum / loss_count
                        progress = (
                            f"epoch {epoch + 1}/{args.epochs}  samples {done_samples:,}/{total_samples:,}  "
                            f"loss {mean_loss:.4f}  {rate:,.0f} samples/s  "
                            f"ETA {format_duration(remaining / rate)}"
                        )
                        sys.stdout.write("\r" + progress.ljust(140))
                        sys.stdout.flush()
                        last_progress = now
                        loss_sum = 0.0
                        loss_count = 0

                    if lines_done >= next_trigger:
                        while lines_done >= next_trigger:
                            next_trigger += cadence
                        eval_loss, eval_offset = evaluate_slice(
                            model, device, pool, eval_path, eval_offset, eval_slice_size
                        )
                        state = make_state(
                            fingerprint, vocab_fingerprint, config, epoch, lines_done,
                            eval_offset, eval_loss, args.lr, done_samples,
                        )
                        path = save_checkpoint(model, optimizer, state)
                        prune_checkpoints()
                        last_saved_global = done_samples
                        print(f"\ncheckpoint {path}  eval loss {eval_loss}")

            if micros_since_step:
                optimizer_step(model, optimizer)
            start_offset = 0
    except KeyboardInterrupt:
        print("\ninterrupted; saving checkpoint")
        state = make_state(
            fingerprint, vocab_fingerprint, config, epoch, lines_done,
            eval_offset, None, args.lr, epoch * train_lines + lines_done,
        )
        if last_saved_global != state["global_samples"]:
            save_checkpoint(model, optimizer, state)
            prune_checkpoints()
        raise SystemExit(130)

    if last_saved_global != done_samples:
        eval_loss, eval_offset = evaluate_slice(
            model, device, pool, eval_path, eval_offset, eval_slice_size
        )
        state = make_state(
            fingerprint, vocab_fingerprint, config, args.epochs, 0,
            eval_offset, eval_loss, args.lr, done_samples,
        )
        path = save_checkpoint(model, optimizer, state)
        prune_checkpoints()
        print(f"\ncheckpoint {path}  eval loss {eval_loss}")
    print("training complete")


if __name__ == "__main__":
    main()

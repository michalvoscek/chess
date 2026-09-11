#!/usr/bin/env python3
"""Predict move probabilities for a position with a trained checkpoint.

Parses the game so far from PGN movetext, rebuilds the current position and
the last five moves, tokenizes them the same way as training data and runs
one forward pass.  The logits are turned into probabilities with softmax
over legal moves only and printed as a JSON array sorted by probability
descending.

Usage:
    python infer.py --elo 1800 --pgn "1. e4 e5 2. Nf3 *"
    python infer.py --elo 1800 --model checkpoints/ckpt_000100000000.pt --device cpu
"""

import argparse
import glob
import io
import json
import os
import sys

import chess
import chess.pgn
import torch

from fen_codec import board_to_sample
from model import (
    ChessTransformer,
    ModelConfig,
    encode_sample,
    legal_move_ids,
    load_vocab,
)
from process_lichess_db import HISTORY_SIZE, MAX_ELO_TOKEN, parse_elo, padded_history
from train import CHECKPOINT_DIR, latest_checkpoint_path, resolve_device

VOCAB_PATH = "vocab.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict probabilities for legal moves")
    parser.add_argument("--elo", required=True, help="elo of the player to move (rounded to nearest 100)")
    parser.add_argument("--pgn", default="", help="game so far in PGN movetext format (default: start position)")
    parser.add_argument("--model", help="checkpoint file or filename fragment in checkpoints/ (default: latest)")
    parser.add_argument("--device", default="gpu:0", help="cpu or gpu:N (default: gpu:0)")
    return parser.parse_args(argv)


def parse_game(pgn_text):
    """Parse PGN movetext into (final board, canonical SAN move list)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        if pgn_text.strip():
            sys.exit("error: could not parse PGN")
        game = chess.pgn.Game()
    if game.errors:
        sys.exit(f"error: invalid PGN: {game.errors[0]}")
    board = game.board()
    history = []
    for move in game.mainline_moves():
        history.append(board.san(move))
        board.push(move)
    return board, history


def resolve_checkpoint_path(value):
    if value is None:
        path = latest_checkpoint_path()
        if path is None:
            sys.exit(f"error: no checkpoint found in {CHECKPOINT_DIR}/")
        return path
    if os.path.isfile(value):
        return value
    matches = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, f"ckpt_*{value}*.pt")))
    if matches:
        return matches[-1]
    sys.exit(f"error: no checkpoint file or prefix match: {value}")


def load_model(path, vocab_fingerprint, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    if state["vocab_fingerprint"] != vocab_fingerprint:
        sys.exit("error: vocab.json changed since the checkpoint was written")
    model = ChessTransformer(ModelConfig.from_dict(state["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = resolve_device(args.device)

    vocab = load_vocab(VOCAB_PATH)
    token_to_id = vocab["token_to_id"]

    elo = parse_elo(args.elo)
    if elo is None:
        sys.exit(f"error: invalid elo, expected integer in 0..{MAX_ELO_TOKEN}: {args.elo}")

    board, history = parse_game(args.pgn)

    sample = board_to_sample(board)
    sample["elo"] = elo
    sample["history"] = padded_history(history[-HISTORY_SIZE:])

    ids = encode_sample(sample, token_to_id)
    if ids is None:
        sys.exit("error: position cannot be tokenized (out-of-range clock or move number)")

    try:
        legal_ids = legal_move_ids(sample, token_to_id)
    except KeyError as error:
        sys.exit(f"error: no vocabulary token for legal move {error.args[0]}")
    legal_sans = [board.san(move) for move in board.legal_moves]

    model = load_model(resolve_checkpoint_path(args.model), vocab["fingerprint"], device)

    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens)[0]
    probs = torch.softmax(logits[legal_ids].float(), dim=0)

    moves = [
        {"move": san, "p": prob}
        for san, prob in sorted(zip(legal_sans, probs.tolist()), key=lambda pair: pair[1], reverse=True)
    ]
    print(json.dumps(moves))


if __name__ == "__main__":
    main()

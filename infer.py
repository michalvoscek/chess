#!/usr/bin/env python3
"""Predict move probabilities for a position with a trained checkpoint.

Parses the game so far from PGN movetext, rebuilds the current position and
the last five moves, runs one forward pass and turns the logits into
probabilities with softmax over legal moves only.  Moves are printed in SAN
(sorted by probability descending) so callers can keep using SAN.

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
import numpy as np
import torch

from fen_codec import HISTORY_SIZE, RECORD_DTYPE, board_to_fields, elo_index, uci_to_san
from move_vocab import NONE_MOVE, move_id_of_uci
from model import ChessTransformer, ModelConfig, make_batch
from train import CHECKPOINT_DIR, latest_checkpoint_path, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict probabilities for legal moves")
    parser.add_argument("--elo", required=True, help="elo of the player to move (rounded to nearest 100)")
    parser.add_argument("--pgn", default="", help="game so far in PGN movetext format (default: start position)")
    parser.add_argument("--model", help="checkpoint file or filename fragment in checkpoints/ (default: latest)")
    parser.add_argument("--device", default="gpu:0", help="cpu or gpu:N (default: gpu:0)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature (default: 1.0; higher plays more loosely)",
    )
    return parser.parse_args(argv)


def parse_game(pgn_text):
    """Parse PGN movetext into (final board, UCI id history)."""
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
        history.append(move)
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


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ChessTransformer(ModelConfig.from_dict(payload["state"]["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def build_record(board, elo, history_moves):
    squares, stm, castling, ep = board_to_fields(board)
    tail = history_moves[-HISTORY_SIZE:]
    history_ids = [NONE_MOVE] * (HISTORY_SIZE - len(tail)) + [
        move_id_of_uci(move.uci()) for move in tail
    ]
    record = np.zeros(1, dtype=RECORD_DTYPE)
    record["squares"][0] = squares
    record["elo"][0] = elo
    record["stm"][0] = stm
    record["castling"][0] = castling
    record["ep"][0] = ep
    record["history"][0] = history_ids
    return record


def main():
    args = parse_args()
    if args.temperature <= 0:
        sys.exit("error: --temperature must be positive")
    device = resolve_device(args.device)

    elo = elo_index(args.elo)
    if elo is None:
        sys.exit(f"error: invalid elo, expected integer in 0..4000: {args.elo}")

    board, history_moves = parse_game(args.pgn)
    record = build_record(board, elo, history_moves)
    legal_sans = [uci_to_san(board, move.uci()) for move in board.legal_moves]
    legal_ids = [move_id_of_uci(move.uci()) for move in board.legal_moves]

    model = load_model(resolve_checkpoint_path(args.model), device)

    batch = make_batch(record, device)
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch)[0]
    probs = torch.softmax(logits[legal_ids].float() / args.temperature, dim=0)

    moves = [
        {"move": san, "p": prob}
        for san, prob in sorted(zip(legal_sans, probs.tolist()), key=lambda pair: pair[1], reverse=True)
    ]
    print(json.dumps(moves))


if __name__ == "__main__":
    main()

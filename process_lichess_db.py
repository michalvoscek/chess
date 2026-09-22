#!/usr/bin/env python3
"""Create per-move training and evaluation samples from a Lichess PGN database.

Usage:
    python process_lichess_db.py --max-games 1000000
    python process_lichess_db.py --max-games 1000000 --input path/to/games.pgn.zst

Writes binary shards into data/:

    train.bin / eval.bin                 RECORD_DTYPE records (see fen_codec.py)
    train.legal.bin / eval.legal.bin     uint16 flat legal-move id lists
    train.legal_off.bin / eval.legal_off.bin   uint32 CSR offsets (n+1)
    meta.json                            counts and fingerprint

Games are assigned in input order: the first 90 percent of valid games go to
training and the remaining 10 percent to evaluation.  The legal move id list
of every position is precomputed so the training loop never needs python-chess.
"""

import argparse
from collections import deque
import glob
import hashlib
import io
import json
import math
import os
import sys
import time

import chess
import chess.pgn
import numpy as np
import zstandard

from fen_codec import HISTORY_SIZE, RECORD_DTYPE, board_to_fields, elo_index
from move_vocab import NONE_MOVE, move_vocab

FLUSH_SAMPLES = 20_000
PROGRESS_INTERVAL = 2.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create per-move train and eval data from a Lichess PGN database"
    )
    parser.add_argument(
        "--max-games",
        required=True,
        type=positive_int,
        help="maximum number of valid games written across both output files",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="source .pgn.zst file (default: latest matching file in data/)",
    )
    return parser.parse_args(argv)


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def get_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def find_latest_input(data_dir):
    pattern = os.path.join(data_dir, "lichess_db_standard_rated_*.pgn.zst")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"no Lichess database found in {data_dir}; use --input to specify one"
        )
    return max(matches)


def promo_char(move):
    return "" if move.promotion is None else chess.piece_symbol(move.promotion)


def create_game_records(game, white_elo, black_elo, move_to_id):
    """One (record, legal_ids) pair per mainline move of a game."""
    board = game.board()
    history = deque([NONE_MOVE] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    records = []

    for move in game.mainline_moves():
        squares, stm, castling, ep = board_to_fields(board)
        target = move_to_id[(move.from_square, move.to_square, promo_char(move))]
        legal = [
            move_to_id[(legal.from_square, legal.to_square, promo_char(legal))]
            for legal in board.legal_moves
        ]
        player_elo = white_elo if board.turn == chess.WHITE else black_elo
        records.append(
            ((tuple(squares), player_elo, stm, castling, ep, target, tuple(history)), legal)
        )
        board.push(move)
        history.append(target)

    return records


class Split:
    """Appends records to a .bin file and legal move ids to a CSR pair."""

    def __init__(self, data_dir, stem):
        self.stem = stem
        self.rec_path = os.path.join(data_dir, f"{stem}.bin")
        self.legal_path = os.path.join(data_dir, f"{stem}.legal.bin")
        self.off_path = os.path.join(data_dir, f"{stem}.legal_off.bin")
        self.rec_file = open(self.rec_path, "wb")
        self.legal_file = open(self.legal_path, "wb")
        self.off_file = open(self.off_path, "wb")
        np.array([0], dtype=np.uint32).tofile(self.off_file)
        self.samples = 0
        self.legal_count = 0
        self.records = []
        self.legal_ids = []
        self.offsets = []

    def add(self, record, legal_ids):
        self.records.append(record)
        self.legal_ids.extend(legal_ids)
        self.legal_count += len(legal_ids)
        self.offsets.append(self.legal_count)
        self.samples += 1
        if len(self.records) >= FLUSH_SAMPLES:
            self.flush()

    def flush(self):
        if self.records:
            np.array(self.records, dtype=RECORD_DTYPE).tofile(self.rec_file)
            self.records.clear()
        if self.legal_ids:
            np.array(self.legal_ids, dtype=np.uint16).tofile(self.legal_file)
            self.legal_ids.clear()
        if self.offsets:
            np.array(self.offsets, dtype=np.uint32).tofile(self.off_file)
            self.offsets.clear()

    def close(self):
        self.flush()
        self.rec_file.close()
        self.legal_file.close()
        self.off_file.close()
        return {
            "samples": self.samples,
            "legal_ids": self.legal_count,
            "record_bytes": os.path.getsize(self.rec_path),
            "legal_bytes": os.path.getsize(self.legal_path),
            "offset_bytes": os.path.getsize(self.off_path),
        }


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_progress(games, max_games, samples, skipped, started_at):
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = games / elapsed
    remaining = max_games - games
    eta = remaining / rate if rate > 0 else 0
    line = (
        f"games {games:,}/{max_games:,}  "
        f"samples {samples:,}  "
        f"skipped {skipped:,}  {rate:,.0f} games/s  ETA {format_duration(eta)}"
    )
    sys.stdout.write("\r" + line.ljust(120))
    sys.stdout.flush()


def main():
    args = parse_args()
    data_dir = get_data_dir()
    input_path = args.input_path or find_latest_input(data_dir)
    if not os.path.isfile(input_path):
        sys.exit(f"error: input file not found: {input_path}")

    _, move_to_id = move_vocab()
    os.makedirs(data_dir, exist_ok=True)
    print(f"processing {input_path}")

    train_split = Split(data_dir, "train")
    eval_split = Split(data_dir, "eval")
    train_game_limit = math.floor(args.max_games * 0.9)

    processed_games = 0
    train_games = 0
    eval_games = 0
    skipped_games = 0
    samples = 0
    started_at = time.monotonic()
    last_progress = started_at
    progress_written = False

    try:
        with open(input_path, "rb") as compressed_file:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed_file) as decompressed_file:
                with io.TextIOWrapper(decompressed_file, encoding="utf-8") as pgn_file:
                    while processed_games < args.max_games:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        white_elo = elo_index(game.headers.get("WhiteElo"))
                        black_elo = elo_index(game.headers.get("BlackElo"))
                        if white_elo is None or black_elo is None or game.errors:
                            skipped_games += 1
                            continue

                        try:
                            game_records = create_game_records(
                                game, white_elo, black_elo, move_to_id
                            )
                        except (ValueError, AssertionError, IndexError, KeyError):
                            skipped_games += 1
                            continue

                        split = train_split if processed_games < train_game_limit else eval_split
                        for record, legal in game_records:
                            split.add(record, legal)
                        if split is train_split:
                            train_games += 1
                        else:
                            eval_games += 1
                        samples += len(game_records)
                        processed_games += 1

                        now = time.monotonic()
                        if now - last_progress >= PROGRESS_INTERVAL:
                            render_progress(
                                processed_games,
                                args.max_games,
                                samples,
                                skipped_games,
                                started_at,
                            )
                            last_progress = now
                            progress_written = True
    except KeyboardInterrupt:
        print()
        print("interrupted; partial output was kept")
        raise SystemExit(130)
    finally:
        train_meta = train_split.close()
        eval_meta = eval_split.close()
        fingerprint = hashlib.sha256(
            f"{train_meta}|{eval_meta}|{train_games}|{eval_games}".encode("utf-8")
        ).hexdigest()
        meta = {
            "format": 1,
            "record_bytes": RECORD_DTYPE.itemsize,
            "none_move": NONE_MOVE,
            "train_games": train_games,
            "eval_games": eval_games,
            "skipped_games": skipped_games,
            "train": train_meta,
            "eval": eval_meta,
            "fingerprint": fingerprint,
        }
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)
            file.write("\n")

    if progress_written:
        print()
    print(
        f"processed {processed_games:,} games "
        f"({train_games:,} train / {eval_games:,} eval); "
        f"skipped {skipped_games:,}"
    )
    print(f"wrote {train_meta['samples']:,} train and {eval_meta['samples']:,} eval samples")
    print(f"wrote {os.path.join(data_dir, 'meta.json')}")


if __name__ == "__main__":
    main()

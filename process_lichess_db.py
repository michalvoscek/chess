#!/usr/bin/env python3
"""Create per-move training and evaluation samples from a Lichess PGN database.

Usage:
    python process_lichess_db.py --max-games 1000000
    python process_lichess_db.py --max-games 1000000 --input path/to/games.pgn.zst

Writes data/train.jsonl and data/eval.jsonl. Games are assigned in input order,
with the first 90 percent going to training and the remaining 10 percent to
evaluation.
"""

import argparse
from collections import deque
import glob
import io
import json
import math
import os
import sys
import time

import chess
import chess.pgn
import zstandard


HISTORY_SIZE = 5
NONE_TOKEN = "<NONE>"
MAX_ELO_TOKEN = 4000
FLUSH_LINES = 100_000
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


def elo_bracket(elo):
    """Round a non-negative integer ELO to the nearest 100, half up."""
    return 100 * ((elo + 50) // 100)


def parse_elo(value):
    if value is None:
        return None

    try:
        elo = int(value.strip())
    except (AttributeError, ValueError):
        return None

    if elo < 0:
        return None

    bracket = elo_bracket(elo)
    if bracket > MAX_ELO_TOKEN:
        return None
    return bracket


def padded_history(previous_moves):
    history = list(previous_moves)
    if len(history) < HISTORY_SIZE:
        history = [NONE_TOKEN] * (HISTORY_SIZE - len(history)) + history
    return history


def create_game_samples(game, white_elo, black_elo):
    board = game.board()
    previous_moves = deque(maxlen=HISTORY_SIZE)
    samples = []

    for move in game.mainline_moves():
        san = board.san(move)
        player_elo = white_elo if board.turn == chess.WHITE else black_elo
        sample = {
            "elo": player_elo,
            "fen": board.fen(),
            "history": padded_history(previous_moves),
            "move": san,
        }
        samples.append(json.dumps(sample, separators=(",", ":")) + "\n")
        board.push(move)
        previous_moves.append(san)

    return samples


def flush_buffer(file_handle, buffer):
    if buffer:
        file_handle.write("".join(buffer))
        buffer.clear()


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_progress(games, max_games, samples, train_samples, eval_samples, skipped, started_at):
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = games / elapsed
    remaining = max_games - games
    eta = remaining / rate if rate > 0 else 0
    line = (
        f"games {games:,}/{max_games:,}  "
        f"samples {samples:,} (train {train_samples:,} / eval {eval_samples:,})  "
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

    train_path = os.path.join(data_dir, "train.jsonl")
    eval_path = os.path.join(data_dir, "eval.jsonl")
    train_game_limit = math.floor(args.max_games * 0.9)

    os.makedirs(data_dir, exist_ok=True)
    print(f"processing {input_path}")
    print(f"writing train data to {train_path}")
    print(f"writing eval data to {eval_path}")

    train_buffer = []
    eval_buffer = []
    processed_games = 0
    train_games = 0
    eval_games = 0
    skipped_games = 0
    train_samples = 0
    eval_samples = 0
    started_at = time.monotonic()
    last_progress = started_at
    progress_written = False

    train_file = None
    eval_file = None
    try:
        train_file = open(train_path, "w", encoding="utf-8")
        eval_file = open(eval_path, "w", encoding="utf-8")

        with open(input_path, "rb") as compressed_file:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed_file) as decompressed_file:
                with io.TextIOWrapper(decompressed_file, encoding="utf-8") as pgn_file:
                    while processed_games < args.max_games:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        white_elo = parse_elo(game.headers.get("WhiteElo"))
                        black_elo = parse_elo(game.headers.get("BlackElo"))
                        if white_elo is None or black_elo is None or game.errors:
                            skipped_games += 1
                            continue

                        try:
                            game_samples = create_game_samples(
                                game, white_elo, black_elo
                            )
                        except (ValueError, AssertionError, IndexError):
                            skipped_games += 1
                            continue

                        if processed_games < train_game_limit:
                            train_buffer.extend(game_samples)
                            train_samples += len(game_samples)
                            train_games += 1
                        else:
                            eval_buffer.extend(game_samples)
                            eval_samples += len(game_samples)
                            eval_games += 1
                        processed_games += 1

                        if len(train_buffer) >= FLUSH_LINES:
                            flush_buffer(train_file, train_buffer)
                        if len(eval_buffer) >= FLUSH_LINES:
                            flush_buffer(eval_file, eval_buffer)

                        now = time.monotonic()
                        if now - last_progress >= PROGRESS_INTERVAL:
                            render_progress(
                                processed_games,
                                args.max_games,
                                train_samples + eval_samples,
                                train_samples,
                                eval_samples,
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
        if train_file is not None:
            flush_buffer(train_file, train_buffer)
            train_file.close()
        if eval_file is not None:
            flush_buffer(eval_file, eval_buffer)
            eval_file.close()

    if progress_written:
        print()
    print(
        f"processed {processed_games:,} games "
        f"({train_games:,} train / {eval_games:,} eval); "
        f"skipped {skipped_games:,}"
    )
    print(f"wrote {train_samples:,} samples to {train_path}")
    print(f"wrote {eval_samples:,} samples to {eval_path}")


if __name__ == "__main__":
    main()

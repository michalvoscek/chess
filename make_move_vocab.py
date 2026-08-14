#!/usr/bin/env python3
"""Generate the complete fixed chess vocabulary and write data/vocab.json.

The move tokens cover every SAN string that any legal chess move can
produce: all 4,084 possible (piece, from, to, promotion) moves, spelled in
every disambiguation form (Nd2, Nbd2, N1d2, Nb1d2) with and without the
capture 'x', each with '', '+' and '#' suffixes. The rest of the tokens
cover every other value used by the 78-token input layout (ELO brackets,
squares, castling flags, side to move, en passant, clocks, <NONE>).

The JSON is flat: "tokens" is an ordered list where the index is the token
id, and "token_to_id" is the reverse lookup. The ordering is deterministic
(first occurrence wins when values are shared between segments, e.g. an
en passant target square and a pawn move spelling), so regenerating the
file always produces identical bytes.

Usage:
    python make_move_vocab.py
    python make_move_vocab.py --verify
"""

import argparse
import glob
import hashlib
import json
import os

FILES = "abcdefgh"
RANKS = "12345678"
PROMOTIONS = "QRBN"
SUFFIXES = ("", "+", "#")
PIECE_LETTERS = "KQRBN"


def get_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def square(index):
    return FILES[index % 8] + RANKS[index // 8]


def all_moves():
    """All (piece, from, to, promotion) moves legal in some position."""
    moves = set()
    for frm in range(64):
        fa, fr = frm % 8, frm // 8
        for to in range(64):
            if frm == to:
                continue
            ta, tr = to % 8, to // 8
            df, dr = ta - fa, tr - fr
            kinds = []
            if abs(df) == abs(dr):
                kinds += ["B", "Q"]
            if df == 0 or dr == 0:
                kinds += ["R", "Q"]
            if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
                kinds += ["N"]
            if abs(df) <= 1 and abs(dr) <= 1:
                kinds += ["K"]
            for kind in kinds:
                moves.add((kind, frm, to, None))

    moves.update([("K", 4, 6, None), ("K", 4, 2, None), ("K", 60, 62, None), ("K", 60, 58, None)])

    for frm in range(64):
        fr = frm // 8
        if fr in (0, 7):
            continue
        fa = frm % 8
        for step, double_rank, promo_rank in ((1, 1, 6), (-1, 6, 1)):
            to = frm + 8 * step
            if fr == promo_rank:
                for promo in PROMOTIONS:
                    moves.add(("P", frm, to, promo))
                    if fa > 0:
                        moves.add(("P", frm, to - 1, promo))
                    if fa < 7:
                        moves.add(("P", frm, to + 1, promo))
            else:
                moves.add(("P", frm, to, None))
                if fr == double_rank:
                    moves.add(("P", frm, frm + 16 * step, None))
                if fa > 0:
                    moves.add(("P", frm, to - 1, None))
                if fa < 7:
                    moves.add(("P", frm, to + 1, None))
    return moves


def san_forms(piece, frm, to, promo):
    """Every SAN spelling of one move, without check/mate suffixes."""
    dest = square(to)
    if piece == "P":
        if promo is not None:
            if to % 8 != frm % 8:
                return [f"{FILES[frm % 8]}x{dest}={promo}"]
            return [f"{dest}={promo}"]
        if to % 8 != frm % 8:
            return [f"{FILES[frm % 8]}x{dest}"]
        return [dest]
    if piece == "K" and (frm, to) in ((4, 6), (4, 2), (60, 62), (60, 58)):
        return ["O-O" if to in (6, 62) else "O-O-O"]
    base = [
        f"{piece}{dest}",
        f"{piece}{FILES[frm % 8]}{dest}",
        f"{piece}{RANKS[frm // 8]}{dest}",
        f"{piece}{FILES[frm % 8]}{RANKS[frm // 8]}{dest}",
    ]
    captures = [s[: s.index(dest)] + "x" + dest for s in base]
    return base + captures


def build_move_tokens():
    moves = all_moves()
    by_piece = {}
    for piece, frm, to, promo in moves:
        by_piece[piece] = by_piece.get(piece, 0) + 1
    tokens = set()
    for piece, frm, to, promo in moves:
        for form in san_forms(piece, frm, to, promo):
            for suffix in SUFFIXES:
                tokens.add(form + suffix)
    return tokens, by_piece


def build_tokens(move_tokens):
    segments = (
        [f"<PLAYER_ELO_{elo}>" for elo in range(0, 4001, 100)],
        ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p", "."],
        ["<TRUE>", "<FALSE>"],
        ["w", "b"],
        ["-"] + [square(i) for i in range(64)],
        [str(i) for i in range(201)],
        [str(i) for i in range(301)],
        ["<NONE>"],
        sorted(move_tokens),
    )
    tokens = []
    seen = set()
    for segment in segments:
        for token in segment:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def build_doc(move_tokens):
    tokens = build_tokens(move_tokens)
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    doc = {
        "version": 1,
        "generated_by": "make_move_vocab.py",
        "total_tokens": len(tokens),
        "tokens": tokens,
        "token_to_id": token_to_id,
    }
    canonical = json.dumps(doc, indent=2, ensure_ascii=False)
    doc["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return doc


def verify_move_coverage(move_tokens, max_games, time_limit):
    import io
    import time

    import chess
    import chess.pgn
    import zstandard

    pattern = os.path.join(get_data_dir(), "lichess_db_standard_rated_*.pgn.zst")
    matches = glob.glob(pattern)
    if not matches:
        print("no lichess database found for verification; skipping scan")
        return
    input_path = max(matches)
    vocab = set(move_tokens)
    observed = set()
    games = 0
    started_at = time.monotonic()

    with open(input_path, "rb") as compressed_file:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as decompressed_file:
            with io.TextIOWrapper(decompressed_file, encoding="utf-8") as pgn_file:
                while games < max_games and time.monotonic() - started_at < time_limit:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        observed.add(board.san(move))
                        board.push(move)
                    games += 1

    missing = observed - vocab
    print(f"verified {games:,} games, {len(observed):,} unique SAN tokens observed")
    print(f"observed moves missing from vocab: {len(missing)}")
    for token in sorted(missing)[:10]:
        print(f"  missing: {token}")
    return not missing


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="vocab.json",
        help="output file (default: vocab.json)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="replay lichess games and check every observed SAN token is in the vocab",
    )
    parser.add_argument(
        "--verify-games",
        type=int,
        default=40_000,
        help="maximum games to scan during --verify (default: 40000)",
    )
    parser.add_argument(
        "--verify-time",
        type=float,
        default=100.0,
        help="seconds budget for --verify scan (default: 100)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    move_tokens, by_piece = build_move_tokens()
    doc = build_doc(move_tokens)

    total_moves = sum(by_piece.values())
    print("moves enumerated:", f"{total_moves:,}", end=" (")
    print(", ".join(f"{piece} {by_piece[piece]:,}" for piece in PIECE_LETTERS) + ", P " + f"{by_piece['P']:,})")
    print("move tokens:", f"{len(move_tokens):,}")
    print("total tokens:", f"{doc['total_tokens']:,}")

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")

    regenerated = json.dumps(build_doc(move_tokens), indent=2, ensure_ascii=False) + "\n"
    with open(args.output, "r", encoding="utf-8") as file:
        if file.read() == regenerated:
            print("determinism check: passed")
        else:
            print("determinism check: FAILED")
            return 1

    if args.verify:
        ok = verify_move_coverage(move_tokens, args.verify_games, args.verify_time)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

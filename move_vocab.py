#!/usr/bin/env python3
"""Deterministic UCI move vocabulary for chess move prediction.

A move is (from_square, to_square, promotion) with a1=0 .. h8=63 and
promotion one of "", "n", "b", "r", "q".  Castling is the king's from-to
(e1g1, e1c1, e8g8, e8c8).  1,968 ids cover every move that is legal in
some position.

Id N_MOVES (1968) is the history padding token <NONE>.  It is never a
legal move and never a training target.

moves.json holds the ordered UCI strings plus a fingerprint so checkpoints
can detect a vocabulary change.

Usage:
    python move_vocab.py
"""

import hashlib
import json

FILES = "abcdefgh"
RANKS = "12345678"
PROMOTIONS = ("", "b", "n", "q", "r")
NONE_MOVE = 1968
N_MOVES = 1968
MOVE_VOCAB_SIZE = N_MOVES + 1
MOVES_PATH = "moves.json"


def square_name(index):
    return FILES[index % 8] + RANKS[index // 8]


def uci_of(frm, to, promo):
    return square_name(frm) + square_name(to) + promo


def all_move_tuples():
    """Every (from, to, promo) legal in some position."""
    moves = set()
    for frm in range(64):
        fa, fr = frm % 8, frm // 8
        for to in range(64):
            if frm == to:
                continue
            ta, tr = to % 8, to // 8
            df, dr = ta - fa, tr - fr
            adf, adr = abs(df), abs(dr)
            if adf == adr or df == 0 or dr == 0:
                moves.add((frm, to, ""))
            if (adf, adr) in ((1, 2), (2, 1)) or (adf <= 1 and adr <= 1):
                moves.add((frm, to, ""))
    moves.update([(4, 6, ""), (4, 2, ""), (60, 62, ""), (60, 58, "")])

    for frm in range(64):
        fr, fa = frm // 8, frm % 8
        if fr in (0, 7):
            continue
        for step, promo_rank, start_rank in ((1, 6, 1), (-1, 1, 6)):
            to = frm + 8 * step
            if fr == promo_rank:
                for promo in PROMOTIONS:
                    if promo == "":
                        continue
                    moves.add((frm, to, promo))
                    if fa > 0:
                        moves.add((frm, to - 1, promo))
                    if fa < 7:
                        moves.add((frm, to + 1, promo))
            else:
                moves.add((frm, to, ""))
                if fr == start_rank:
                    moves.add((frm, frm + 16 * step, ""))
                if fa > 0:
                    moves.add((frm, to - 1, ""))
                if fa < 7:
                    moves.add((frm, to + 1, ""))
    return moves


def build_doc():
    ordered = sorted(all_move_tuples())
    assert len(ordered) == N_MOVES, len(ordered)
    ucis = [uci_of(*move) for move in ordered]
    doc = {
        "version": 1,
        "generated_by": "move_vocab.py",
        "total_moves": N_MOVES,
        "none_move": NONE_MOVE,
        "moves": ucis,
    }
    canonical = json.dumps(doc, indent=2, ensure_ascii=False)
    doc["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return doc, {move: index for index, move in enumerate(ordered)}


_DOC = None
_MOVE_TO_ID = None


def move_vocab():
    """Load moves.json once and return (doc, move_to_id)."""
    global _DOC, _MOVE_TO_ID
    if _DOC is None:
        with open(MOVES_PATH, "r", encoding="utf-8") as file:
            _DOC = json.load(file)
        _MOVE_TO_ID = {
            (frm, to, promo): index
            for index, (frm, to, promo) in enumerate(
                sorted(all_move_tuples())
            )
        }
        assert _DOC["total_moves"] == N_MOVES
    return _DOC, _MOVE_TO_ID


def move_id_of(frm, to, promo):
    return move_vocab()[1][(frm, to, promo)]


def move_id_of_uci(uci):
    frm = (int(uci[1]) - 1) * 8 + FILES.index(uci[0])
    to = (int(uci[3]) - 1) * 8 + FILES.index(uci[2])
    promo = uci[4:] if len(uci) > 4 else ""
    return move_id_of(frm, to, promo)


def uci_of_id(move_id):
    if move_id == NONE_MOVE:
        return "<NONE>"
    return move_vocab()[0]["moves"][move_id]


def _self_test():
    doc, move_to_id = build_doc()
    assert doc["total_moves"] == N_MOVES
    assert doc["none_move"] == NONE_MOVE
    assert len(doc["moves"]) == N_MOVES
    assert len(set(doc["moves"])) == N_MOVES
    assert len(move_to_id) == N_MOVES
    for move, index in move_to_id.items():
        assert uci_of(*move) == doc["moves"][index]
    assert (4, 6, "") in move_to_id
    assert (60, 58, "") in move_to_id
    assert (12, 28, "") in move_to_id  # e2e4
    assert (52, 60, "q") in move_to_id  # e7e8q
    assert move_id_of(12, 28, "") == move_id_of_uci("e2e4")
    assert uci_of_id(move_id_of_uci("e7e8q")) == "e7e8q"
    assert uci_of_id(move_id_of_uci("e1g1")) == "e1g1"
    print(f"self test passed; {N_MOVES} moves, fingerprint {doc['fingerprint'][:16]}")


def main():
    doc, _ = build_doc()
    with open(MOVES_PATH, "w", encoding="utf-8") as file:
        file.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"moves: {doc['total_moves']}")
    print(f"wrote {MOVES_PATH}")
    regenerated, _ = build_doc()
    with open(MOVES_PATH, "r", encoding="utf-8") as file:
        if file.read() == json.dumps(regenerated, indent=2, ensure_ascii=False) + "\n":
            print("determinism check: passed")
        else:
            print("determinism check: FAILED")
            return 1
    _self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

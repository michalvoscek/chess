#!/usr/bin/env python3
"""Convert between chess positions and the compact training record fields.

Each train/eval sample stores one position as:

    squares[64]  uint8   0 empty, 1-6 white PNBRQK, 7-12 black pnbrqk (a1=0 .. h8=63)
    stm          uint8   0 white, 1 black
    castling     uint8   bit0 wk, bit1 wq, bit2 bk, bit3 bq
    ep           uint8   0 none, 1-64 = en passant target square index + 1

and, alongside the position fields:

    elo          uint8   Elo bracket / 100 (0..40)
    target       uint16  UCI move id (see move_vocab.py)
    history[5]   uint16  previous UCI move ids, front-padded with <NONE>

Usage:
    from fen_codec import board_to_fields, fields_to_board, uci_to_san
"""

import chess
import numpy as np

N_PIECE = 13
N_STM = 2
N_CASTLING = 16
N_EP = 65
N_ELO = 41
MAX_ELO = 4000
HISTORY_SIZE = 5
RECORD_FIELDS = ("squares", "elo", "stm", "castling", "ep", "target", "history")
RECORD_DTYPE = np.dtype(
    [
        ("squares", np.uint8, (64,)),
        ("elo", np.uint8),
        ("stm", np.uint8),
        ("castling", np.uint8),
        ("ep", np.uint8),
        ("target", np.uint16),
        ("history", np.uint16, (HISTORY_SIZE,)),
    ],
    align=False,
)
assert RECORD_DTYPE.itemsize == 80, RECORD_DTYPE.itemsize
SYMBOLS = ".PNBRQKpnbrqk"
CASTLING_BITS = (
    (chess.BB_H1, 1),
    (chess.BB_A1, 2),
    (chess.BB_H8, 4),
    (chess.BB_A8, 8),
)


def piece_code(piece):
    if piece is None:
        return 0
    return piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6


def code_symbol(code):
    return SYMBOLS[code]


def elo_bracket(elo):
    """Round a non-negative integer Elo to the nearest 100, half up."""
    return 100 * ((elo + 50) // 100)


def elo_index(value):
    """Map an Elo value or bracket to its embedding index 0..40, or None."""
    if value is None:
        return None
    try:
        elo = int(value)
    except (TypeError, ValueError):
        return None
    if elo < 0:
        return None
    bracket = elo_bracket(elo)
    if bracket > MAX_ELO:
        return None
    return bracket // 100


def board_to_fields(board):
    """Position fields of a record (squares, stm, castling, ep)."""
    squares = [piece_code(board.piece_at(square)) for square in range(64)]
    castling = 0
    for mask, bit in CASTLING_BITS:
        if board.castling_rights & mask:
            castling |= bit
    ep_square = board.ep_square
    ep = 0 if ep_square is None else ep_square + 1
    return squares, (0 if board.turn == chess.WHITE else 1), castling, ep


def _compress_row(row):
    parts = []
    dots = 0
    for char in row:
        if char == ".":
            dots += 1
        else:
            if dots:
                parts.append(str(dots))
                dots = 0
            parts.append(char)
    if dots:
        parts.append(str(dots))
    return "".join(parts)


def fields_to_fen(squares, stm, castling, ep):
    """Rebuild the FEN string stored in record position fields."""
    rows = []
    for rank in range(7, -1, -1):
        rows.append(_compress_row("".join(code_symbol(squares[rank * 8 + file]) for file in range(8))))
    castling_flags = ""
    for letter, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)):
        if castling & bit:
            castling_flags += letter
    ep_name = "-" if ep == 0 else chess.square_name(ep - 1)
    return " ".join(("/".join(rows), "w" if stm == 0 else "b", castling_flags or "-", ep_name, "0", "1"))


def fields_to_board(squares, stm, castling, ep):
    """Rebuild the board stored in record position fields."""
    return chess.Board(fields_to_fen(squares, stm, castling, ep))


def uci_to_san(board, uci):
    return board.san(chess.Move.from_uci(uci))


def _self_test():
    positions = [
        chess.Board(),
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 3"),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 1"),
        chess.Board("8/8/8/8/8/8/8/K6k w - - 37 92"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ]
    ep_board = chess.Board()
    ep_board.push_san("e4")
    positions.append(ep_board)
    a3_board = chess.Board()
    a3_board.push_san("a4")
    positions.append(a3_board)
    for board in positions:
        squares, stm, castling, ep = board_to_fields(board)
        assert len(squares) == 64
        assert 0 <= stm < N_STM and 0 <= castling < N_CASTLING and 0 <= ep < N_EP
        rebuilt = fields_to_board(squares, stm, castling, ep)
        assert rebuilt.piece_map() == board.piece_map(), (board.fen(), rebuilt.fen())
        assert rebuilt.turn == board.turn
        assert rebuilt.castling_rights == board.castling_rights, (board.fen(), rebuilt.fen())
    assert elo_index(1500) == 15
    assert elo_index(1540) == 15
    assert elo_index(1550) == 16
    assert elo_index(4049) == 40
    assert elo_index(4050) is None
    assert elo_index("nope") is None
    print(f"self test passed for {len(positions)} positions")


if __name__ == "__main__":
    _self_test()

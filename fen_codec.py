#!/usr/bin/env python3
"""Convert between chess positions and the sample fields used in train/eval data.

Training data stores each position as flat fields instead of FEN.  This
module converts in both directions:

* board -> sample fields, used when generating train/eval data
  (process_lichess_db.py)
* sample fields -> board, used for legal move masking during training
  and for inference

It never maps to or from token ids; vocabulary lookup belongs to the
scripts that consume the samples.

Usage:
    from fen_codec import board_to_sample, sample_to_fen, sample_to_board
"""

import chess


def board_to_sample(board):
    """Convert a board to the position-related fields of a train/eval sample."""
    position = []
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            position.append(piece.symbol() if piece else ".")
    castling = {
        "castling_wk": "<TRUE>" if board.has_kingside_castling_rights(chess.WHITE) else "<FALSE>",
        "castling_wq": "<TRUE>" if board.has_queenside_castling_rights(chess.WHITE) else "<FALSE>",
        "castling_bk": "<TRUE>" if board.has_kingside_castling_rights(chess.BLACK) else "<FALSE>",
        "castling_bq": "<TRUE>" if board.has_queenside_castling_rights(chess.BLACK) else "<FALSE>",
    }
    ep_square = board.ep_square
    return {
        "position": "".join(position),
        **castling,
        "side_to_move": "w" if board.turn == chess.WHITE else "b",
        "en_passant": chess.square_name(ep_square) if ep_square else "-",
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
    }


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


def sample_to_fen(sample):
    """Rebuild the FEN string of the position stored in a sample."""
    rows = [sample["position"][i * 8 : (i + 1) * 8] for i in range(8)]
    placement = "/".join(_compress_row(row) for row in rows)
    castling = ""
    for flag, letter in (
        ("castling_wk", "K"),
        ("castling_wq", "Q"),
        ("castling_bk", "k"),
        ("castling_bq", "q"),
    ):
        if sample[flag] == "<TRUE>":
            castling += letter
    castling = castling or "-"
    return " ".join(
        (
            placement,
            sample["side_to_move"],
            castling,
            sample["en_passant"],
            str(sample["halfmove_clock"]),
            str(sample["fullmove_number"]),
        )
    )


def sample_to_board(sample):
    """Rebuild the board stored in a sample."""
    return chess.Board(sample_to_fen(sample))


def _self_test():
    positions = [
        chess.Board(),
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 3"),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 1"),
        chess.Board("8/8/8/8/8/8/8/K6k w - - 37 92"),
    ]
    ep_board = chess.Board()
    ep_board.push_san("e4")
    positions.append(ep_board)
    sample_fields = {
        "position",
        "castling_wk",
        "castling_wq",
        "castling_bk",
        "castling_bq",
        "side_to_move",
        "en_passant",
        "halfmove_clock",
        "fullmove_number",
    }
    for board in positions:
        sample = board_to_sample(board)
        assert set(sample) == sample_fields
        rebuilt = sample_to_board(sample)
        assert rebuilt.fen() == board.fen(), (board.fen(), rebuilt.fen())
    print(f"self test passed for {len(positions)} positions")


if __name__ == "__main__":
    _self_test()

# Training data properties produced by this module:
#
#     position            yes (from board)
#     castling_wk         yes (from board)
#     castling_wq         yes (from board)
#     castling_bk         yes (from board)
#     castling_bq         yes (from board)
#     side_to_move        yes (from board)
#     en_passant          yes (from board)
#     halfmove_clock      yes (from board)
#     fullmove_number     yes (from board)
#     elo                 no  (from game headers, player to move)
#     history             no  (from previous moves of the game)
#     move                no  (target move of the sample, from the game)
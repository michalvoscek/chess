#!/usr/bin/env python3
"""Chess move prediction model: record batching and the transformer.

The transformer reads a 74-token sequence: Elo, 64 square tokens (piece
embedding plus learned square embedding), side to move, castling,
en passant, five history moves, and a learned <QUERY> token.  Attention
is bidirectional (the whole position is observed at once).  The QUERY
hidden state is projected through the tied move embedding to produce
logits over the 1,968 UCI moves plus the <NONE> history pad.

Input and output share only the move embedding (a history "e2e4" and the
output "e2e4" are the same concept).  State fields get their own small
tables so tokens cannot collide.

Usage:
    from model import (
        SEQ_LEN,
        ModelConfig,
        ChessTransformer,
        legal_move_loss,
        make_batch,
    )
"""

import numpy as np
import torch
from torch import nn

from fen_codec import (
    HISTORY_SIZE,
    N_CASTLING,
    N_ELO,
    N_EP,
    N_PIECE,
    N_STM,
    RECORD_DTYPE,
)
from move_vocab import MOVE_VOCAB_SIZE

SEQ_LEN = 74
N_SQUARE = 64


class ModelConfig:
    def __init__(
        self,
        dim=512,
        layers=8,
        heads=8,
        ffn=2048,
        dropout=0.1,
        seq_len=SEQ_LEN,
        vocab_size=MOVE_VOCAB_SIZE,
    ):
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.ffn = ffn
        self.dropout = dropout
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def as_dict(self):
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def make_batch(records, device):
    """Turn a RECORD_DTYPE array into the tensor dict ChessTransformer expects."""
    if not isinstance(records, np.ndarray):
        records = np.array(records, dtype=RECORD_DTYPE)
    return {
        "squares": torch.as_tensor(records["squares"], dtype=torch.long, device=device),
        "elo": torch.as_tensor(records["elo"], dtype=torch.long, device=device),
        "stm": torch.as_tensor(records["stm"], dtype=torch.long, device=device),
        "castling": torch.as_tensor(records["castling"], dtype=torch.long, device=device),
        "ep": torch.as_tensor(records["ep"], dtype=torch.long, device=device),
        "history": torch.as_tensor(records["history"], dtype=torch.long, device=device),
        "targets": torch.as_tensor(records["target"], dtype=torch.long, device=device),
    }


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)
        self.attn = nn.MultiheadAttention(
            config.dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.RMSNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.ffn),
            nn.GELU(),
            nn.Linear(config.ffn, config.dim),
        )

    def forward(self, x):
        hidden = self.norm1(x)
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ChessTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        self.elo_emb = nn.Embedding(N_ELO, dim)
        self.piece_emb = nn.Embedding(N_PIECE, dim)
        self.square_emb = nn.Embedding(N_SQUARE, dim)
        self.stm_emb = nn.Embedding(N_STM, dim)
        self.castling_emb = nn.Embedding(N_CASTLING, dim)
        self.ep_emb = nn.Embedding(N_EP, dim)
        self.move_emb = nn.Embedding(config.vocab_size, dim)
        self.query = nn.Parameter(torch.empty(dim))
        self.pos_emb = nn.Embedding(config.seq_len, dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(dim)
        self.apply(_init_weights)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, batch):
        batch_size = batch["squares"].size(0)
        squares = self.piece_emb(batch["squares"]) + self.square_emb.weight.unsqueeze(0)
        x = torch.cat(
            [
                self.elo_emb(batch["elo"]).unsqueeze(1),
                squares,
                self.stm_emb(batch["stm"]).unsqueeze(1),
                self.castling_emb(batch["castling"]).unsqueeze(1),
                self.ep_emb(batch["ep"]).unsqueeze(1),
                self.move_emb(batch["history"]),
                self.query.view(1, 1, -1).expand(batch_size, 1, -1),
            ],
            dim=1,
        )
        x = self.dropout(x + self.pos_emb.weight.unsqueeze(0))
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, -1]) @ self.move_emb.weight.T


def legal_move_loss(logits, legal_ids, target_ids):
    """Cross entropy with softmax restricted to each sample's legal moves.

    logits: [batch, vocab] tensor.
    legal_ids: list of per-sample token id lists.
    target_ids: LongTensor [batch] of played move token ids.
    """
    logits = logits.float()
    batch, width = logits.size(0), max(len(ids) for ids in legal_ids)
    index = torch.zeros((batch, width), dtype=torch.long)
    mask = torch.zeros((batch, width), dtype=torch.bool)
    for row, ids in enumerate(legal_ids):
        index[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = True
    index, mask = index.to(logits.device), mask.to(logits.device)
    gathered = logits.gather(1, index).masked_fill(~mask, float("-inf"))
    target_logits = logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return (torch.logsumexp(gathered, dim=1) - target_logits).mean()


def _self_test():
    config = ModelConfig(dim=64, layers=2, heads=4, ffn=128, dropout=0.0)
    model = ChessTransformer(config)
    batch_size = 3
    records = np.zeros(batch_size, dtype=RECORD_DTYPE)
    records["elo"] = [0, 15, 40]
    records["target"] = [0, 1, 2]
    records["history"] = MOVE_VOCAB_SIZE - 1
    batch = make_batch(records, torch.device("cpu"))
    logits = model(batch)
    assert logits.shape == (batch_size, MOVE_VOCAB_SIZE)

    legal = [[0, 1, 2], [3, 4], list(range(10))]
    targets = torch.tensor([0, 3, 5])
    loss = legal_move_loss(logits, legal, targets)
    loss.backward()
    assert torch.isfinite(loss)

    full = ChessTransformer(ModelConfig())
    params = sum(p.numel() for p in full.parameters())
    assert SEQ_LEN == 1 + N_SQUARE + 1 + 1 + 1 + HISTORY_SIZE + 1
    print(f"self test passed; full model parameters: {params:,}")


if __name__ == "__main__":
    _self_test()

#!/usr/bin/env python3
"""Chess move prediction model: sample tokenization and the transformer.

Tokenization turns a train/eval sample into the fixed 78-token id sequence
described in README.md plus the target move token id.  The model is a
transformer over that sequence with bidirectional attention: the whole
position is observed at once, so there is no causal mask.  The final
hidden states are mean pooled and projected through the tied token
embedding to produce logits.

Samples whose values have no token (out-of-range clocks or move numbers,
unknown moves) tokenize to None and are skipped by callers, per README.

Usage:
    from model import (
        SEQ_LEN,
        ModelConfig,
        ChessTransformer,
        encode_sample,
        legal_move_ids,
        legal_move_loss,
        load_vocab,
        tokenize_sample,
    )
"""

import json

import torch
from torch import nn

from fen_codec import sample_to_board

SEQ_LEN = 78


def load_vocab(path):
    """Load vocab.json and return the document with tokens and token_to_id."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def encode_sample(sample, token_to_id):
    """Map a train/eval sample to its input token ids, or None."""
    lookup = token_to_id.get
    position = sample["position"]
    history = sample["history"]
    if len(position) != 64 or len(history) != 5:
        return None
    values = (
        [f"<PLAYER_ELO_{sample['elo']}>"]
        + list(position)
        + [sample["castling_wk"], sample["castling_wq"], sample["castling_bk"], sample["castling_bq"]]
        + [sample["side_to_move"], sample["en_passant"], str(sample["halfmove_clock"]), str(sample["fullmove_number"])]
        + list(history)
    )
    ids = [lookup(value) for value in values]
    if None in ids:
        return None
    return ids


def tokenize_sample(sample, token_to_id):
    """Map a train/eval sample to (input_ids, target_id), or None."""
    target = token_to_id.get(sample["move"])
    if target is None:
        return None
    ids = encode_sample(sample, token_to_id)
    if ids is None:
        return None
    return ids, target


def legal_move_ids(sample, token_to_id):
    """Token ids of every legal move in the sample's position."""
    board = sample_to_board(sample)
    return [token_to_id[board.san(move)] for move in board.legal_moves]


class ModelConfig:
    def __init__(
        self,
        vocab_size,
        seq_len=SEQ_LEN,
        dim=512,
        layers=12,
        heads=8,
        ffn=2048,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.ffn = ffn
        self.dropout = dropout

    def as_dict(self):
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.normal_(module.in_proj_weight, std=0.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


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
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.seq_len, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.dim)
        self.apply(_init_weights)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        pooled = self.norm(x).mean(dim=1)
        return pooled @ self.token_embedding.weight.T


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
    vocab = load_vocab("vocab.json")
    token_to_id = vocab["token_to_id"]
    with open("data/train.jsonl", "r", encoding="utf-8") as file:
        sample = json.loads(file.readline())
    tokenized = tokenize_sample(sample, token_to_id)
    assert tokenized is not None, "first train sample failed to tokenize"
    ids, target = tokenized
    assert len(ids) == SEQ_LEN
    legal = legal_move_ids(sample, token_to_id)
    assert target in legal, "played move not among legal moves"

    config = ModelConfig(
        vocab_size=vocab["total_tokens"], dim=64, layers=2, heads=4, ffn=128, dropout=0.0
    )
    model = ChessTransformer(config)
    logits = model(torch.tensor([ids, ids]))
    assert logits.shape == (2, vocab["total_tokens"])
    loss = legal_move_loss(logits, [legal, legal], torch.tensor([target, target]))
    loss.backward()
    assert torch.isfinite(loss)

    with torch.device("meta"):
        full = ChessTransformer(ModelConfig(vocab_size=vocab["total_tokens"]))
    params = sum(p.numel() for p in full.parameters())
    print(f"self test passed; full model parameters: {params:,}")


if __name__ == "__main__":
    _self_test()

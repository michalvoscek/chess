# Chess AI

Goal of the project is to take input ELO and generate moves like real human with that elo would do.  
Use python venv.  

## Input and output

Application should accept input:
1. elo of player to move
2. current position
3. last 5 moves (in PGN format)

and return output:
1. probabilities for all legal moves in given position

Moves are predicted as UCI (`e2e4`, `e7e8q`, `e1g1`) and reported as SAN so
callers keep working with SAN.

## Scripts

### Download
Download database from lichess

```bash
python download_lichess_db.py --month 2026-05
```

Should save data to `data/lichess_db_standard_rated_2026-05.pgn.zst`  
Files can be large verify available disk size first.  
Download should be resumable.  
Download progress should be displayed on standard output.  

### Train data
decompress game files
```bash
python process_lichess_db.py --max-games 1000000
```

Creates one sample for every move.  Output is binary shards in `data/`
(see `fen_codec.py` for the 80-byte record layout):

| File | Contents |
|---|---|
| `train.bin` / `eval.bin` | `RECORD_DTYPE` records (squares, elo, stm, castling, ep, target, history) |
| `train.legal.bin` / `eval.legal.bin` | uint16 flat legal-move id lists |
| `train.legal_off.bin` / `eval.legal_off.bin` | uint32 CSR offsets (`n+1`) |
| `meta.json` | counts and fingerprint |

Games are split 90/10 in input order (first 90 percent → train).  Games
missing Elo (or with PGN errors) are skipped.  Elo is rounded to the nearest
100 and stored as an index `0..40`.  The legal move id list of every
position is precomputed so training never needs `python-chess`.

args:
--max-games: limit games in output files
--input: source `.pgn.zst` file

### Generate move vocabulary

```bash
python move_vocab.py
```

Writes `moves.json`: the 1,968 UCI moves (from-to-promo) legal in some
position, plus a fingerprint.  Id `1968` is the history pad token `<NONE>`.

## Tokenization

Input is a fixed 74-token sequence.  Each field has its own embedding table
so tokens cannot collide (the old shared vocab made `b` mean both "black to
move" and "black bishop").

| Segment | Values | Count | Embedding |
|---|---|---|---|
| elo | index 0–40 (Elo/100) | 1 | 41 |
| squares, a1→h8 | 13 (`0`, `PNBRQK`, `pnbrqk`) | 64 | piece 13 + square 64 |
| side to move | 0/1 | 1 | 2 |
| castling | 4-bit mask | 1 | 16 |
| en passant | 0 = none, else square+1 | 1 | 65 |
| history | UCI move ids, front-`<NONE>` padded | 5 | move table |
| `<QUERY>` | learned readout token | 1 | parameter |
| **Total** | | **74 tokens, always** | |

Output is a logit per UCI move (plus `<NONE>`, never legal and never a
target), taken as `h_query @ move_emb.T`.  History and output share the move
embedding because they denote the same concept.

If a value has no token (out-of-range Elo), the sample is skipped for
training and an error is shown for inference.

## Train

```bash
python train.py
```

Shuffles the sample order every epoch (a batch of 256 consecutive moves
comes from ~6 whole games, which makes gradients useless without shuffling).
Effective batch size 512 via gradient accumulation 2×256.

args:
--epochs: n - how many times to go thru training data  
--lr: r - peak learning rate (default 0.0003); warmup 2000 steps then cosine decay to 10%  
--wd: AdamW weight decay on weights and embeddings (default 0.1)  
--batch-size: n (default 512)  
--max-samples: cap training samples per epoch (default: all)  
--device: cpu or gpu:1 what device to use for training  
--resume: continue from the latest checkpoint in `checkpoints/`  
--checkpoint-cadence: after how many samples a checkpoint is created  

### Optimizer

AdamW

Weight decay        0.1 (applied to weights + embeddings only; biases and RMSNorm gains excluded)
Peak learning rate  3e-4 (linear warmup 2000 steps, then cosine to 3e-5)
Gradient clipping   1

### Model

Decoder-only Transformer over the 74-token sequence

Layers:                                8
Hidden Size (embedding dimension):     512
Attention Heads:                       8
FFN Size (4× hidden size):             2048
Context:                               74 tokens
Parameters:                            ~26M
Positional encoding:                   learned embeddings (74 × 512 table, trained with the model)
Normalization                          pre-norm + RMSNorm
Move vocabulary                        1,968 UCI moves (`moves.json`)
Activation                             GELU
Dropout                                0.1
Precision                              bf16

Readout is the `<QUERY>` hidden state; the output projection is tied to the
history move embedding.

### Resuming after data changes

When `data/train.bin` is replaced with new games, continue training from the
old weights instead of random init.  `train.py` keeps the weights but resets
the sample stream when `data/meta.json`'s fingerprint changed.

### Loss

Cross entropy with softmax thru legal moves in vocabulary only (precomputed
in the shards).  Only the move contributes (position fields get no target).

## Evaluate

```bash
python evaluate.py
```

Reports on `data/eval.bin`:

1. **Move-match** top-1 / top-3 / CE against held-out human moves, bucketed
   by the player's Elo.  Humans agree with other humans on roughly 40–55%
   of moves (top-1) — that is the reference band.
2. **Cross-Elo CE matrix** — the same positions re-evaluated with the Elo
   token overwritten.  A visible diagonal means the model uses Elo.
3. **Elo KL probe** — mean `KL(p_low || p_high)` over legal moves.  Near
   zero means the model ignores Elo.

args:
--checkpoint: checkpoint file (default: latest)
--samples / --cross-elo-samples / --kl-samples: sample counts
--device / --batch-size

## Inference

Program should interpret game so far, calculate current position and last 5 moves.  
Forward pass produces logits; softmax over legal moves gives probabilities.

```bash
python infer.py --elo 1800 --pgn "1. e4 e5 2. Nf3 *"
```

flags:  
--elo: target elo of the player to move rounded to 100.  
--pgn: game so far in pgn format empty on first move  
--model: checkpoint file (in `.pt` format), by default latest is used based on `checkpoints/ckpt_*.pt` search  
--device: `cpu` or `gpu<n>` default is `gpu:0`  
--temperature: softmax temperature (default 1.0; higher plays more loosely)

Output is JSON with probabilities in SAN:

```json
[{"move": "e5", "p": 0.418}, {"move": "Nf6", "p": 0.305}]
```

## GUI

Simple GUI written in react using chessground.js library to visualize and input position and moves.  
Should have one screen at start containing:
1. standard chess board in opening position
2. input for elo (allow any number but should be rounded to closest available token for expressing elo), elo can change during game
3. input for temperature (1 = human-like sampling, higher = looser)
4. button that plays next move (no matter what color has turn computer can play both both sides), move played should be based on probability it recieves from inference of the model
5. buttons to restart game and undo last move
6. list of moves played in pgn format

Board shouls allow user to make only legal moves.  
Promotions should be handled by picker dialog.  
Inference should be used by spawning new process for `infer.py` script.  

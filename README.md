# Chess AI

Goal of the project is to take input ELO and generate moves like real human with that elo would do.  
Use python venv.  

## Input and output

Application should accept input:
1. elo of game
2. current position (in FEN format)
3. last 5 moves (in PGN format)

and return output:
1. probabilities for all legal moves in given position

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

Instead of one sample per game, create one sample for every move.

Game:
```
e4
e5
Nf3
Nc6
Bb5
a6
Ba4
```

becomes

Input
```
<PLAYER_ELO_1800>

rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

<NONE> <NONE> <NONE> <NONE> <NONE>
```

Output

```
e4
```

Input

```
<PLAYER_ELO_1800>

rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

<NONE> <NONE> <NONE> <NONE> e4
```

Output

```
e5
```

Input
```
<PLAYER_ELO_1800>

rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2

<NONE> <NONE> <NONE> e4 e5

```
Output

```
Nf3
```

### Tokenization

For FEN position use per character tokenization.

For moves use one token per move.

Good examples
```
e4
Nf3
Bb5
O-O
Qxe5+
```

Avoid character-level tokenization.

Special tokens:

```
<NONE>
<PLAYER_ELO_1200>
```

## Train

### Optimizer

AdamW

Weight decay     0.1

### Model
Decoder-only Transformer

Layers:          12
Hidden Size:     512
Attention Heads: 8
FFN Size:        2048
Context:         512 tokens
Parameters:      ~45–60M

```bash
python train.py
```
flags:
--epochs: n - how many times to go thru training data  
--lr: r - default 0.0003, how much should training affect weights  
--device: cpu or gpu:1 what device to use for training  

### Resuming after data changes

When `data/train.jsonl` is replaced with new games, continue training from
the old weights instead of random init.

Important: It might be needed to extend move set vocabulary for new moves not seen in previous training data.  
Store vocab with checkpoint.  
Keep existing IDs stable.  
Initialize new weights at random.  
Build vocab before training from train.jsonl.  

For ELO there should be be predefined tokens from `<PLAYER_ELO_0>` to `<PLAYER_ELO_4000>` incrementing by 100. So extending is not needed here.

For training data fingerprint should be calculated so using sha256 of `train.jsonl`

```bash
python train.py --epochs 1 --lr 3e-4
```

--resume: initialize latest checkpoint based on training data fingerprint and offset in train samples stream automatically saved;
--checkpoint-cadence: number, after how many samples should checkpoint be created

## Inference

Program should interpret game so far, calculate current position and last 5 moves.  
Then for every legal move it should run thru model to calculate probability of that move.

```bash
python infer.py --elo 1800 --pgn "1. e4 e5 2. Nf3 *"
```

flags:  
--elo: target elo of the player to move  
--pgn: game so far in pgn format  
--model: checkpoint file or prefix (default: latest)  

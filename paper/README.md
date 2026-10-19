# Paper

LaTeX write-up of the project: data pipeline, tokenization, model,
training, evaluation, and current training results.

Build the PDF (requires `pdflatex` and `latexmk`):

```bash
latexmk -pdf paper.tex
```

Output is `paper.pdf`. `loss_curve.png` is a copy of
`plots/loss_curve.png` (regenerate with `python plot_loss.py` and
re-copy if training continues).

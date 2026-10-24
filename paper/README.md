ArXiv preprint (ICML 2025 style file used for layout only; not an accepted ICML paper).

Compile with:
```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Generate figures (from repo root):
```bash
python paper/scripts/plot_hero.py              # fig:teaser (intro)
python paper/scripts/plot_teaser.py            # fig:refinement (§3.2)
python paper/scripts/plot_shared_fractional.py
python paper/scripts/plot_horizon_ablation.py
python paper/scripts/plot_bilinear_norm.py
```

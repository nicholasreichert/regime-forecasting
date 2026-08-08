#!/usr/bin/env bash
# Build the paper. MiKTeX on this machine installs outside PATH, so add the
# usual per-user location if pdflatex is not already resolvable.
set -e
command -v pdflatex >/dev/null 2>&1 || \
  export PATH="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64:$PATH"

cd "$(dirname "$0")"
python -m src.experiments.fit_tables 2>/dev/null || true   # run from repo root instead
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "built: $(pwd)/main.pdf"

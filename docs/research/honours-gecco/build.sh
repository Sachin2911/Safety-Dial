#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
build_dir=../../../outputs/honours-gecco/build
mkdir -p "$build_dir"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" honoursThesis.tex
source_dir="$PWD"
(cd "$build_dir" && BIBINPUTS="$source_dir:" bibtex honoursThesis)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" honoursThesis.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" honoursThesis.tex
cp "$build_dir/honoursThesis.pdf" honoursThesis.pdf

#!/usr/bin/env bash
# Compile the Safety Dial pilot report.
# LaTeX sources live in ./latex; intermediates stay in ./latex/build;
# the final PDF lands here as ./SafeDialReport.pdf.
#
# Usage:
#   ./compile.sh          build the PDF
#   ./compile.sh clean    remove the build directory and the compiled PDF
set -euo pipefail

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="$DOC_DIR/latex"
BUILD_DIR="$LATEX_DIR/build"
OUT_PDF="$DOC_DIR/SafeDialReport.pdf"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD_DIR" "$OUT_PDF"
    echo "Cleaned build artifacts."
    exit 0
fi

cd "$LATEX_DIR"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD_DIR" main.tex

cp "$BUILD_DIR/main.pdf" "$OUT_PDF"
echo "PDF written to $OUT_PDF"

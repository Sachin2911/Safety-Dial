#!/usr/bin/env bash
set -euo pipefail

concept_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${concept_dir}"
exec latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

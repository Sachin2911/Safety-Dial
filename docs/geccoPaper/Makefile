.PHONY: all figures pictures clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
figures:
	cd ../.. && uv run python experiments/helpers/geccoPaper.py
pictures:
	cd ../.. && uv run python experiments/helpers/geccoPaper.py --render
clean:
	latexmk -c main.tex

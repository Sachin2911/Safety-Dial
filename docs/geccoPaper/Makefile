.PHONY: all named anonymous figures pictures clean
all: named
named:
	latexmk -pdf -interaction=nonstopmode -halt-on-error named.tex
anonymous:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
figures:
	cd ../.. && uv run python experiments/helpers/geccoPaper.py
pictures:
	cd ../.. && uv run python experiments/helpers/geccoPaper.py --render
clean:
	latexmk -c main.tex named.tex

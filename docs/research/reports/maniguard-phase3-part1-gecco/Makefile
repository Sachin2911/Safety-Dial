TECTONIC ?= tectonic
.PHONY: all clean
all: named.pdf
named.pdf: main.tex named.tex references.bib
	$(TECTONIC) -k --keep-logs named.tex
clean:
	rm -f named.aux named.bbl named.blg named.log named.out comment.cut

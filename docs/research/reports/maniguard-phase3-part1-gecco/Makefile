TECTONIC ?= tectonic
.PHONY: all checkin clean
all: named.pdf
checkin: draft.pdf meeting-brief.pdf
draft.pdf: draft.tex references.bib $(wildcard ../../figures/ManiGuard/phase3-recovery/*.pdf)
	$(TECTONIC) -k --keep-logs draft.tex
meeting-brief.pdf: meeting-brief.tex $(wildcard ../../figures/ManiGuard/phase3-recovery/*.pdf)
	$(TECTONIC) -k --keep-logs meeting-brief.tex
named.pdf: main.tex named.tex references.bib followup-methods.tex part2-results.tex part3-results.tex part4-results.tex $(wildcard ../../figures/ManiGuard/phase3-part4/*.pdf)
	$(TECTONIC) -k --keep-logs named.tex
clean:
	rm -f named.aux named.bbl named.blg named.log named.out comment.cut

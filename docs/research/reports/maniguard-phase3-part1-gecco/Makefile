TECTONIC ?= tectonic
.PHONY: all paper checkin clean
all: draft.pdf
paper: draft.pdf
checkin: draft.pdf meeting-brief.pdf
draft.pdf: draft.tex native-noise-paper.tex model-followup.tex references.bib latent-references.bib \
  $(wildcard ../../figures/ManiGuard/latent-evolution/*.pdf) \
  $(wildcard ../../figures/ManiGuard/continuation-screen-20261020/*.pdf)
	$(TECTONIC) -k --keep-logs draft.tex
meeting-brief.pdf: meeting-brief.tex $(wildcard ../../figures/ManiGuard/phase3-recovery/*.pdf)
	$(TECTONIC) -k --keep-logs meeting-brief.tex
named.pdf: main.tex named.tex references.bib followup-methods.tex part2-results.tex part3-results.tex part4-results.tex $(wildcard ../../figures/ManiGuard/phase3-part4/*.pdf)
	$(TECTONIC) -k --keep-logs named.tex
clean:
	rm -f named.aux named.bbl named.blg named.log named.out comment.cut

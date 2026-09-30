# Builds the PDF from main.tex (needs tectonic: brew install tectonic). The figure PDFs in figures/ are
# generated locally from private results and are not distributed, so this only works where they exist.
main.pdf: main.tex
	tectonic -X compile main.tex

clean:
	rm -f main.pdf

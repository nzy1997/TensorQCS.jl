JL = julia --project=.
JLE = julia --project=examples

init:
	$(JL) -e 'using Pkg; Pkg.instantiate()'

resetshor:
	$(JLE) examples/resetshorcode.jl

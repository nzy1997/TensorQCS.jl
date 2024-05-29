module TensorQCS
using Yao
using OMEinsum
using LinearAlgebra
using Yao.YaoBlocks.Optimise

# qc2ein
export ComplexConj, SymbolRecorder, ein_circ,ConnectMap, qc2enisum

# coerror
export coherent_error_unitary, error_quantum_circuit,toput

include("qc2ein.jl")
include("coerror.jl")
end

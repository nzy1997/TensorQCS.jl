module TensorQCS

using Yao
using OMEinsum
using LinearAlgebra
using Yao.YaoBlocks.Optimise
using CUDA
# qc2ein
export ComplexConj, SymbolRecorder,IdentityRecorder, ein_circ, ConnectMap, qc2enisum

# coerror
export coherent_error_unitary, error_quantum_circuit,toput, error_pairs,add_indentity

# shorcode
export do_circuit_simulation

include("qc2ein.jl")
include("coerror.jl")
include("shorcodereset.jl")
end

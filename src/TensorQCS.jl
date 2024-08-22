module TensorQCS

using Yao
using LinearAlgebra
using TensorQEC.Yao.YaoBlocks.Optimise
using CUDA
using TensorQEC
using Combinatorics

# shorcode
export do_circuit_simulation,classical_decode,print_state,error_probabillity

# circuits
export reset_shor_circuit,reset_shor_code_total

# threshold
export double_error_location,check_double_pos

include("shorcodereset.jl")
include("circuits.jl")
include("threshold.jl")
end

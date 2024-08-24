using TensorQCS
using TensorQCS.TensorQEC
using TensorQCS.TensorQEC.Yao
using DelimitedFiles
using Random
using TensorQCS.CUDA
using Test
using LinearAlgebra
using Yao.YaoBlocks.Optimise

CUDA.allowscalar(true)
CUDA.device!(1)


qc,qcen = reset_shor_code_total()

qcn,qc_locs,qubit_locs = double_error_location(qc,10:18,(1:14)∪(52:67),[14,67])

@show check_triple_pos(qcen,qc,qc_locs[1:129],qubit_locs[1:129])

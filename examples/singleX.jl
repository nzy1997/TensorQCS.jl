using TensorQCS
using TensorQCS.Yao
using DelimitedFiles
using TensorQCS.CUDA
CUDA.allowscalar(false)
CUDA.device!(0)

function singleX(exqc)
xinf = []

reg = rand_state(1;nbatch =200)
reg = cu(reg)

reg0 = copy(reg)
infs = Vector{Vector{Float64}}()
for i in 1:500
    apply!(reg, exqc)
    apply!(reg, exqc)
    inf = 1 .- fidelity(reg, reg0)
    @show i, inf[1]
    push!(infs, inf)
end
writedlm("examples/data/Xinfs.csv", infs)
writedlm("examples/data/xvector.csv", xinf)
end
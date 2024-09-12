using TensorQCS
using TensorQCS.TensorQEC
using TensorQCS.TensorQEC.Yao
using DelimitedFiles
using Random
using TensorQCS.CUDA
using Test
using LinearAlgebra
CUDA.allowscalar(false)
CUDA.device!(1)


gate_inf = Float64[]
eps = Float64[]
eps_ec = Float64[]
eps_ec2 = Float64[]
for error_rate in [1e-7,2*1e-7,4*1e-7,8*1e-7,1.6*1e-6,3.2*1e-6,6.4*1e-6,1.28*1e-5,2.56*1e-5,5.12*1e-5,1e-4,2*1e-4,4*1e-4,8*1e-4,1.6*1e-3,3.2*1e-3,6.4*1e-3,1.28*1e-2,2.56*1e-2,5.12*1e-2,1e-1,0.2,0.4]
	for j in 1:100
		@show j,error_rate
		qc, qcen, vector,qcx,eqcz = error_circuit_nocopy(error_rate)
        push!(gate_inf,sum(vector)/length(vector))
		xinfs = singleX(qcx;iters = 1)
		infs = do_circuit_simulation(qc, qcen,eqcz; use_cuda = true, iters = 100,ct=1)
        infs2 = do_circuit_simulation(qc, qcen,eqcz; use_cuda = true, iters = 100,ct=1000)
        push!(eps,xinfs[end])
        push!(eps_ec,infs[end])
        push!(eps_ec2,infs2[end])
	end
end
writedlm("examples/data/data.csv", [gate_inf,eps,eps_ec,eps_ec2])
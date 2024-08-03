using TensorQCS
using TensorQCS.TensorQEC
using TensorQCS.TensorQEC.Yao
using DelimitedFiles
using Random
using TensorQCS.CUDA
using Test
CUDA.allowscalar(false)
CUDA.device!(0)

include("functions.jl")

for error_rate in [1e-3,1e-2,1e-1]
	for j in 1:4
		@show j,error_rate
		qc, qcen, vector,qcx,eqcz = error_circuit(error_rate)
		xinfs = singleX(qcx;iters = 1000)
		writedlm("examples/data/E($error_rate)Xinfs($j).csv", xinfs)
		writedlm("examples/data/E($error_rate)vector($j).csv", vector)
		for ct in [1,10,50,100,2000] 
			infs = do_circuit_simulation(qc, qcen,eqcz; use_cuda = true, iters = 1000,ct)
			writedlm("examples/data/E($error_rate)infs($j)ct($ct).csv", infs)
		end
	end
end

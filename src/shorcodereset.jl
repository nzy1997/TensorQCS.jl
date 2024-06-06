import Yao.fidelity

function fidelity(reg1::CuArray, reg2::CuArray)
	reg1_cpu = reg1 |> cpu 
	reg2_cpu = reg2 |> cpu 
	return fidelity(reg1_cpu, reg2_cpu)
end
function do_circuit_simulation(qc::ChainBlock,qcen::ChainBlock; error_rate = 1e-5, iters = 10,use_cuda = false,nbatch = 3)
	pairs, vector = error_pairs(error_rate)
	eqc = error_quantum_circuit(qc, pairs)
	eqcen = error_quantum_circuit(qcen, pairs)

	regrs = rand_state(1;nbatch)
	reg = join(zero_state(12;nbatch), regrs, zero_state(8;nbatch))
	use_cuda && (reg = reg |> cu)

	reg0 = copy(reg)
	apply!(reg, subroutine(eqcen, 1:9))
	infs = Vector{Vector{Float64}}()
	for i in 1:iters
		apply!(reg, eqc)
		apply!(reg, eqc)
		regt = apply(reg, subroutine(eqcen', 1:9))
		regt = regt |> cpu
		reg0 = reg0 |> cpu
		inf = 1 .- fidelity(regt, reg0)
		@show i, inf
		push!(infs, inf)
	end
    return infs,vector
end


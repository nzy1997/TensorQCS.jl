function do_circuit_simulation(qc::ChainBlock,qcen::ChainBlock,eqcz::ChainBlock;  iters = 10,use_cuda = false,nbatch = 3, ct=1)
	regrs = zero_state(1;nbatch)
	reg = join(zero_state(12;nbatch), regrs, zero_state(8;nbatch))
	use_cuda && (reg = reg |> cu)

	reg0 = copy(reg)
	apply!(reg, subroutine(qcen, 1:9))
	infs = Vector{Vector{Float64}}()
	for i in 1:iters
		# rqc = add_rand_pauli(qc)
		apply!(reg, eqcz)
		apply!(reg, eqcz)
		i%ct ==0 && apply!(reg, qc)
		regt = apply(reg, subroutine(qcen', 1:9))
		inf = 1 .- fidelity(regt, reg0)
		i%10 ==0 && print("i = $i ")
		push!(infs, inf)
		if sum(inf)/nbatch > 0.5
			@show sum(inf)
			@show "break iter:", i
			break
		end
	end
    return infs
end

function add_rand_pauli(qc::ChainBlock)
	nq = nqubits(qc)
	rgatepos = mod1(rand(Int),length(qc))
	rpauli = mod1(rand(Int),2)
	rqubitpos = mod1(rand(Int),nq)
	c = 0
	qcr = chain(nq)
	for gate in qc
		push!(qcr,gate)
		c = c +1
		if c == rgatepos
			push!(qcr,put(nq,rqubitpos =>  (rpauli==1) ? X : Z ))
		end
	end
	return qcr
end

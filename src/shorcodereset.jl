function do_circuit_simulation(qc::ChainBlock,qcen::ChainBlock,eqcz::ChainBlock;  iters = 10,use_cuda = false, ct=1)
	num_qubits = nqubits(qc)
	reg = zero_state(num_qubits)
	use_cuda && (reg = reg |> cu)

	apply!(reg, subroutine(qcen, 1:9))
	erps = Vector{Float64}()

	onevec = [classical_decode(DitStr{2,9}(i)) for i in 0:511]
	for i in 1:iters
		# rqc = add_rand_pauli(qc)
		apply!(reg, eqcz)
		apply!(reg, eqcz)
		i%ct ==0 && apply!(reg, qc)
		push!(erps, error_probabillity(reg,onevec))
		i%10 ==0 && println("i = $i")
	end
    return erps
end

# true or 1 represents |1>
function classical_decode(btc::DitStr{2, 9, Int64})
	return (sum(btc[1:3])>1) ⊻ (sum(btc[4:6])>1) ⊻ (sum(btc[7:9])>1) 
end

function error_probabillity(reg::ArrayReg)
	onevec = [classical_decode(DitStr{2,9}(i)) for i in 0:511]
	return error_probabillity(reg,onevec)
end

function error_probabillity(reg::ArrayReg,onevec::Vector{Bool})
	return sum(abs2.(reg.state[findall(x->x,onevec)]))
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

function print_state(reg; atol = 1e-10)
	println(reg)
	nq = nqubits(reg)
	ids = findall(>(atol), abs2.(reg.state))
	println("non zero bits: $(length(ids))")
	for id in ids
		println("nbatch = $(id.I[2]), bits = $(BitStr{nq}(id.I[1] - 1)), val = $(reg.state[id])")
	end
end
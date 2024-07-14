function do_circuit_simulation(qc::ChainBlock,qcen::ChainBlock,eqcz::ChainBlock;  iters = 10,use_cuda = false, ct=1)
	reg = zero_state(21)
	use_cuda && (reg = reg |> cu)
	apply!(reg, subroutine(qcen, 1:9))
	erps = Vector{Float64}()
	onevec = [classical_decode(DitStr{2,9}(i)) for i in 0:511]
	for i in 1:iters
		# rqc = add_rand_pauli(qc)
		apply!(reg, eqcz)
		apply!(reg, eqcz)
		i%ct ==0 && apply!(reg, qc)
		# print_state(reg)
		push!(erps, error_probabillity(reg,onevec))
		i%10 ==0 && println("i = $i ")
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
	if size(reg.state) == (512,1)
		return sum(abs2.(reg.state[onevec]))
	end
	focus!(reg,1:9)
	ans = sum(abs2.(reg.state[onevec,onevec]))
	relax!(reg)
	return ans
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

notzero(x) = !iszero(x)
function print_state(reg)
	println(reg)
	nq = nqubits(reg)
	ids = findall(isone, notzero.(reg.state))
	println("non zero bits: $(length(ids))")
	for id in ids
		println("nbatch = $(id.I[2]), bits = $(BitStr{nq}(id.I[1] - 1)), val = $(reg.state[id])")
	end
end
using TensorQCS
using TensorQCS.TensorQEC
using TensorQCS.TensorQEC.Yao
using DelimitedFiles
using Random
using TensorQCS.CUDA
using Test
CUDA.allowscalar(true)
CUDA.device!(0)

function meandcr!(qc::ChainBlock, i::Int, st_me, qccr, num_qubits)
	qcme = chain(12)
	TensorQEC.measure_circuit!(qcme, st_me[i], 9 + mod1(i, 3))
	push!(qc, subroutine(qcme, 10:21))
	if mod(i, 3) == 0
		push!(qc, qccr[i-2])
		push!(qc, qccr[i-1])
		push!(qc, qccr[i])
		push!(qc, Measure(num_qubits; locs = 19:21, resetto = bit"000"))
	end
end

function reset_shor_circuit()
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	st_me = stabilizers(ShorCode(), linearly_independent = false)
	num_qubits = 21

	st_pos = [19, 20, 21]
	qccr = chain(
		num_qubits,
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => Z),
		control(num_qubits, (st_pos[1], st_pos[3]), 4 => Z),
		control(num_qubits, (st_pos[2], st_pos[3]), 7 => Z),
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 2 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 3 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 4 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 5 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 6 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 7 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 8 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 9 => X),
	)
	# Z error, X stabilizers
	qc1 = chain(num_qubits)

	[push!(qc1, control(num_qubits,  i, 9+i => X)) for i in 1:9]
	for i in 1:3
		meandcr!(qc1, i, st_me, qccr, num_qubits)
	end
	push!(qc1, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))

	# X error, Z stabilizers
	qc2 = chain(num_qubits)
	[push!(qc2, control(num_qubits, 9+i, i => X)) for i in 1:9]
	for i in 4:12
		meandcr!(qc2, i, st_me, qccr, num_qubits)
	end
	push!(qc2, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))
	qc3 = chain([put(num_qubits, i => X) for i in 1:9]...)
	return qc1,qc2,qc3,qcen
end

function error_circuit(error_rate)
	pairs, vector = error_pairs(error_rate)
	qc1,qc2,qcx,qcen = reset_shor_circuit()
	num_qubits = nqubits(qc1)
	qc = chain(num_qubits)
	push!(qc, put(21, 18 => H))
	push!(qc, subroutine(qcen, 10:18))
	eqc1 = error_quantum_circuit_pair_replace(qc1, pairs)
	push!(qc, eqc1)

	push!(qc, subroutine(qcen, 10:18))
	eqc2 = error_quantum_circuit_pair_replace(qc2, pairs)
	push!(qc, eqc2)

	return qc, qcen, vector, error_quantum_circuit_pair_replace(chain(1,X), pairs),error_quantum_circuit_pair_replace(qcx,pairs)
end


function singleX(exqc;iters = 10)
	reg = zero_state(1)
	# reg = cu(reg)
	erp = Vector{Float64}()
	for i in 1:iters
		apply!(reg, exqc)
		apply!(reg, exqc)
		push!(erp,abs2(reg.state[2]))
		i%10 ==0 && print("i = $i ")
	end
	return erp
end

for error_rate in [1e-5,1e-4,1e-3]
	for j in 1:2
		@show j,error_rate
		qc, qcen, vector,qcx,eqcz = error_circuit(1e-5)
		xinfs = singleX(qcx;iters = 1000)
		writedlm("examples/data/E($error_rate)Xinfs($j).csv", xinfs)
		writedlm("examples/data/E($error_rate)vector($j).csv", vector)
		for ct in [1,10,50,100,2000] 
			infs = do_circuit_simulation(qc, qcen,eqcz; use_cuda = true, iters = 1000,ct)
			writedlm("examples/data/E($error_rate)infs($j)ct($ct).csv", infs)
		end
	end
end

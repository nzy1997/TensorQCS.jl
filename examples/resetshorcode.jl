using TensorQCS
using TensorQEC
using TensorQCS.Yao
using DelimitedFiles
using Random
using TensorQCS.CUDA
using Test
CUDA.allowscalar(false)
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

function reset_shor_circuit(error_rate)
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	qcen = chain(9, put(9, 9 => H), qcen)
	data_qubit_num = size(code.matrix, 2) ÷ 2
	st_me = stabilizers(ShorCode(), linearly_independent = false)
	num_qubits = 21

	pairs, vector = error_pairs(error_rate)

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
	qc = chain(num_qubits)
	push!(qc, put(21, 18 => H))
	push!(qc, subroutine(qcen, 10:18))

	qc1 = chain(num_qubits)

	[push!(qc1, control(num_qubits, 9 + i, i => X)) for i in 1:9]
	for i in 1:3
		meandcr!(qc1, i, st_me, qccr, num_qubits)
	end
	push!(qc1, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))
	eqc1 = error_quantum_circuit(qc1, pairs)
	push!(qc, eqc1)

	# X error, Z stabilizers
	qc2 = chain(num_qubits)
	push!(qc, subroutine(qcen, 10:18))
	[push!(qc2, control(num_qubits, i, 9 + i => X)) for i in 1:9]
	for i in 4:12
		meandcr!(qc2, i, st_me, qccr, num_qubits)
	end
	push!(qc2, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))
	eqc2 = error_quantum_circuit(qc2, pairs)
	push!(qc, eqc2)
	qc3 = chain(put(num_qubits, 1 => Z),put(num_qubits, 4 => Z),put(num_qubits, 7 => Z))
	return qc, qcen, vector, error_quantum_circuit(chain(1,X), pairs),error_quantum_circuit(qc3,pairs)
end

function singleX(exqc,nbatch;iters = 10)
	reg = rand_state(1; nbatch)
	reg = cu(reg)

	reg0 = copy(reg)
	infs = Vector{Vector{Float64}}()
	for i in 1:iters
		apply!(reg, exqc)
		apply!(reg, exqc)
		inf = 1 .- fidelity(reg, reg0)
		i%10 ==0 && print("i = $i ")
		push!(infs, inf)
		if sum(inf)/nbatch > 0.5
			break
		end
	end
	return infs
end
# for error_rate in [1e-8,5*1e-7,1e-7,5*1e-6,1e-6]
# 	for j in 1:8
# 		nbatch = 100
# 		@show j,error_rate
# 		qc, qcen, vector,qcx,eqcz = reset_shor_circuit(error_rate)

# 		xinfs = singleX(qcx,nbatch)
# 		writedlm("examples/data/E($error_rate)Xinfs($j).csv", xinfs)

# 		infs = do_circuit_simulation(qc, qcen; use_cuda = true, iters = 1000, nbatch )
# 		writedlm("examples/data/E($error_rate)infs($j).csv", infs)
# 		writedlm("examples/data/E($error_rate)vector($j).csv", vector)
# 	end
# end

for error_rate in [1e-8]
	qc, qcen, vector,qcx,eqcz = reset_shor_circuit(error_rate)
	for j in [1,5,10,20,50,100]
		nbatch = 100
		@show j,error_rate
		

		xinfs = singleX(qcx,nbatch;iters = 1000)
		writedlm("examples/data/E($error_rate)Xinfs($j).csv", xinfs)

		infs = do_circuit_simulation(qc, qcen,eqcz; use_cuda = true, iters = 1000, nbatch ,ct =j)
		writedlm("examples/data/E($error_rate)infs($j).csv", infs)
		writedlm("examples/data/E($error_rate)vector($j).csv", vector)
	end
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

# infs, vector = do_circuit_simulation(qc, qcen; error_rate= 1e-5, use_cuda = true, iters=500, nbatch=1)

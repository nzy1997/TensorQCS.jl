using TensorQCS
using TensorQEC
using TensorQCS.Yao

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
	qcen = chain(9, put(9, 9 => H), qcen)
	data_qubit_num = size(code.matrix, 2) ÷ 2
	st_me = stabilizers(ShorCode(), linearly_independent = false)
	num_qubits = 21
	qc = chain(num_qubits)

    push!(qc, put(num_qubits, 1 => Z))
    push!(qc, put(num_qubits, 4 => Z))
    push!(qc, put(num_qubits, 7 => Z))

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
	push!(qc, put(21, 18 => H))
	push!(qc, subroutine(qcen, 10:18))
    [push!(qc, control(num_qubits, 9+i, i => X) ) for i in 1:9]
	for i in 1:3
		meandcr!(qc, i, st_me, qccr, num_qubits)
	end
	push!(qc, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))

	# X error, Z stabilizers
	push!(qc, subroutine(qcen, 10:18))
    [push!(qc, control(num_qubits, i, 9+i => X) ) for i in 1:9]
	for i in 4:12
		meandcr!(qc, i, st_me, qccr, num_qubits)
	end
	push!(qc, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))
	return qc, qcen
end

qc, qcen = reset_shor_circuit()
vizcircuit(qc)
pairs,vector = error_pairs(1e-5) 
eqc = error_quantum_circuit(qc,pairs)
eqcen = error_quantum_circuit(qcen,pairs)

regrs = rand_state(1)
reg = join(zero_state(12), regrs, zero_state(8))
infs = []
for i in 1:100
	apply!(reg, subroutine(eqcen, 1:9))
	apply!(reg, eqc)
    apply!(reg, eqc)
	apply!(reg, subroutine(eqcen', 1:9))
    inf = 1 - fidelity(reg, join(zero_state(12), regrs, zero_state(8))) 
	@show i, inf
    push!(infs,inf)
end

# Error X without correction
reg1 = copy(regrs)
infs = []
for i in 1:100000
	apply!(reg1, pairs[1].second)
	apply!(reg1, pairs[1].second)
    inf = 1 - fidelity(reg1, regrs) 
	@show i, inf
    push!(infs,inf)
end

using Test
@testset "reset_shor_circuit" begin
	qc, qcen = reset_shor_circuit()
	regrs = rand_state(1)
	reg = join(zero_state(12), regrs, zero_state(8))
	apply!(reg, subroutine(qcen, 1:9))
	apply!(reg, qc)
	apply!(reg, subroutine(qcen', 1:9))
	@test fidelity(reg, join(zero_state(12), regrs, zero_state(8))) ≈ 1
end

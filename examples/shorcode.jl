using TensorQCS
using TensorQEC
using TensorQCS.Yao
using TensorQCS.Yao.YaoBlocks.Optimise
using TensorQCS.OMEinsum

import TensorQCS.Yao.YaoBlocks.Optimise: to_basictypes

function shor_code_cir()
	st = TensorQEC.stabilizers(ShorCode())
	qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
	qcen = chain(9, put(9, 9 => H), qcen)
	data_qubit_num = size(code.matrix, 2) ÷ 2
	st_me = TensorQEC.stabilizers(ShorCode(), linearly_independent = false)
	qcm, st_pos, num_qubits = measure_circuit(st_me)

	qccr = chain(
		num_qubits,
		control(num_qubits, (10, 11), 1 => Z),
		control(num_qubits, (10, 12), 4 => Z),
		control(num_qubits, (11, 12), 7 => Z),
		control(num_qubits, (13, 14), 1 => X),
		control(num_qubits, (13, 15), 2 => X),
		control(num_qubits, (14, 15), 3 => X),
		control(num_qubits, (16, 17), 4 => X),
		control(num_qubits, (16, 18), 5 => X),
		control(num_qubits, (17, 18), 6 => X),
		control(num_qubits, (19, 20), 7 => X),
		control(num_qubits, (19, 21), 8 => X),
		control(num_qubits, (20, 21), 9 => X),
	)

	qcf = chain(num_qubits)
	push!(qcf, subroutine(num_qubits, qcen, 1:data_qubit_num))
	# for i in 1:codesize
	#     push!(qcf, put(num_qubits, i => X))
	# end 
	push!(qcf, qcm)
	push!(qcf, qccr)
	push!(qcf, subroutine(num_qubits, qcen', 1:data_qubit_num))
	return simplify(qcf; rules = [to_basictypes, Optimise.eliminate_nested]), data_qubits, num_qubits
end

# function error_compare(qc::ChainBlock, error_rate::ChainBlock)
qc, data_qubits, num_qubits = shor_code_cir()
qce, vec = error_quantum_circuit(qc, 1e-5)
qce = chain(num_qubits, put(num_qubits, data_qubits[1] => X), qce)
qc = chain(num_qubits, put(num_qubits, data_qubits[1] => X), qc)

cm = ConnectMap(setdiff(1:num_qubits, data_qubits), data_qubits, num_qubits)
qcf, srs = ein_circ(qce, cm)
tn = qc2enisum(qcf, srs, cm)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
# return contract(optnet)[1]
# end
qce, vec = error_quantum_circuit(qcf, 1e-5)
tne = qc2enisum(qce, srs, cm)
optnet = optimize_code(tne, TreeSA(), OMEinsum.MergeVectors())

YaoPlots.CircuitStyles.r[] = 0.3
vizcircuit(qcf; starting_texts = 1:2*num_qubits, filename = "ToricCode.svg")


function shor_code_play()
	st = TensorQEC.stabilizers(ShorCode())
	qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
	qcen = chain(9, put(9, 9 => H), qcen)

	reg0 = zero_state(9)
	apply!(reg0, qcen)

	reg1 = zero_state(9)
	apply!(reg1, put(9, 9 => X))
	apply!(reg1, qcen)

	fidelity(reg0, reg1)
	regrs = rand_state(1)
	reg = join(regrs, zero_state(8))
	apply!(reg, qcen)

	@show fidelity(reg0, reg)
	@show fidelity(reg1, reg)
	@show abs.(regrs.state)
end

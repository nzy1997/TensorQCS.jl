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
	qcm, st_pos, num_qubits = measure_circuit_steane(qcen,st_me,3)

	qccr = chain(
		num_qubits,
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => Z),
		control(num_qubits, (st_pos[1], st_pos[3]), 4 => Z),
		control(num_qubits, (st_pos[2], st_pos[3]), 7 => Z),
		control(num_qubits, (st_pos[4], st_pos[5]), 1 => X),
		control(num_qubits, (st_pos[4], st_pos[6]), 2 => X),
		control(num_qubits, (st_pos[5], st_pos[6]), 3 => X),
		control(num_qubits, (st_pos[7], st_pos[8]), 4 => X),
		control(num_qubits, (st_pos[7], st_pos[9]), 5 => X),
		control(num_qubits, (st_pos[8], st_pos[9]), 6 => X),
		control(num_qubits, (st_pos[10], st_pos[11]), 7 => X),
		control(num_qubits, (st_pos[10], st_pos[12]), 8 => X),
		control(num_qubits, (st_pos[11], st_pos[12]), 9 => X),
	)

	qcf = chain(num_qubits)
	# push!(qcf, subroutine(num_qubits, qcen, 1:data_qubit_num))

	# push!(qcf, put(num_qubits, 1 => Z))
    # push!(qcf, put(num_qubits, 4 => Z))
    # push!(qcf, put(num_qubits, 7 => Z))

	push!(qcf, qcm)
	push!(qcf, qccr)
	# push!(qcf, subroutine(num_qubits, qcen', 1:data_qubit_num))
	return simplify(qcf; rules = [to_basictypes, Optimise.eliminate_nested]), data_qubits, num_qubits,qcen
end

function error_tensornetwork(qc::ChainBlock, error_rate::Real,num_qubits::Int, data_qubits::Vector{Int})
    qce, vec = error_quantum_circuit(qc, error_rate)
    qce = chain(num_qubits, put(num_qubits, data_qubits[1] => X), qce)

    cm = ConnectMap(data_qubits,setdiff(1:num_qubits, data_qubits), num_qubits)
    qcf, srs = ein_circ(qce, cm)
    tn = qc2enisum(qcf, srs, cm)
    return tn, vec
end

function error_infidelity(error_rates::Vector{Float64})
    qc, data_qubits, num_qubits,qcen = shor_code_cir()
    tn,vec = error_tensornetwork(qc, error_rates[1], num_qubits, data_qubits)
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors()) 
    for error_rate in error_rates
        tn,vec = error_tensornetwork(qc, error_rate, num_qubits, data_qubits)
        # optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
        inf = 1-abs(contract(TensorNetwork(optnet.code,tn.tensors))[1]/4)
        @show inf
        @show vec
    end
end   
error_infidelity([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3])

YaoPlots.CircuitStyles.r[] = 0.3
vizcircuit(qcf; starting_texts = 1:nqubits(qcf), filename = "ToricCode_2.svg")

function two_rounds_circuit()
	qc, data_qubits, num_qubits ,qcen= shor_code_cir()
	cm = ConnectMap(data_qubits,setdiff(1:69, data_qubits), 69)
	qc2 = chain(69,put(69,9=>ConstGate.S),subroutine(69,qcen,1:9),subroutine(69,qc,1:39),subroutine(69,qc,(1:9) ∪ (40:69)),subroutine(69,qcen',1:9))
	qc2 = simplify(qc2; rules = [to_basictypes, Optimise.eliminate_nested])
	qcf, srs = ein_circ(qc2, cm)
	qcf, idrs = add_indentity(qcf, collect(1:39))
	return qcf, srs, idrs,cm
end
qcf, srs, idrs,cm = two_rounds_circuit()
tn = qc2enisum(qcf, srs, cm)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors()) 
inf = 1-abs(contract(optnet)[1]/4)

cost, gradient = OMEinsum.cost_and_gradient(optnet.code, (optnet.tensors...,))

ansvec = [gradient[i.symbol]*optnet.tensors[i.symbol]'-optnet.tensors[i.symbol]*gradient[i.symbol]' for i in idrs]
abssum = (x->sum(abs.(x))).(ansvec)

function show_indices(qc::ChainBlock, srsnum::Int)
	qc = simplify(qc; rules = [to_basictypes, Optimise.eliminate_nested])
	qc2 = TensorQCS.dm_circ(qc::ChainBlock)
	qc2 = simplify(qc2; rules = [to_basictypes, Optimise.eliminate_nested])
	qcn = chain(2*nqubits(qc))
	srs = [SymbolRecorder() for _ in 1:srsnum]
	srscount = 1
	for gate in qc2
		push!(qcn, gate)
		for i in toput(gate).locs
			push!(qcn, put(2*nqubits(qc), i => srs[srscount]))
			srscount += 1
		end
	end
	return qcn,srs
end

qcs, srss = show_indices(qc2, 1200)
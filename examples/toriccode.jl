using TensorQCS
using TensorQEC
using TensorQCS.Yao
using TensorQCS.Yao.YaoBlocks.Optimise
using TensorQCS.OMEinsum

import TensorQCS.Yao.YaoBlocks.Optimise: to_basictypes

function toric_code_cir(codesize::Int)
    t = ToricCode(codesize, codesize)
    st = TensorQEC.stabilizers(t)
    qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
    data_qubit_num = size(code.matrix, 2) ÷ 2
    st_me = TensorQEC.stabilizers(t,linearly_independent = false)
    qcm,st_pos, num_qubits = measure_circuit_fault_tol(st_me)

    bimat = TensorQEC.stabilizers2bimatrix(st_me)
    table = make_table(bimat.matrix, 1)
    qccr = correct_circuit(table, st_pos, num_qubits, data_qubit_num, data_qubit_num)

    qcf = chain(num_qubits)
    push!(qcf,put(num_qubits, data_qubits[1]=>X))
    push!(qcf, subroutine(num_qubits, qcen, 1:data_qubit_num))
    for i in 1:codesize
        push!(qcf, put(num_qubits, i => X))
    end 
    push!(qcf,qcm)
    push!(qcf,qccr)
    push!(qcf, subroutine(num_qubits, qcen', 1:data_qubit_num))
    return simplify(qcf; rules=[to_basictypes, Optimise.eliminate_nested]), data_qubits, num_qubits
end
qc, data_qubits, num_qubits = toric_code_cir(2)

cm = ConnectMap(data_qubits, setdiff(1:num_qubits,data_qubits), 2)
qcf, srs = ein_circ(qc, cm)
tn = qc2enisum(qcf, srs, cm)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)[1]
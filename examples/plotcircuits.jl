using TensorQEC
using TensorQCS.Yao

include("functions.jl")

qc1,qc2,qc3,qcen = reset_shor_circuit()

Yao.CircuitStyles.r[] = 0.3
vizcircuit(qc1[1:end-2]; filename = "qc1.pdf")
vizcircuit(qc2[1:end-2];filename = "qc2.pdf")

function bit_flip_rep_code()
    st = [paulistring(3,4,[1,2]), paulistring(3,4,[1,3]),paulistring(3,4,[2,3])]
    qcbf = chain(9)
    qcbf1 = chain(9)
    qcbf2 = chain(9)
    [push!(qcbf1, control(9, i,i+3=>X)) for i in 1:3]
    qcme,pos = TensorQEC.measure_circuit(st)
    push!(qcbf, qcbf1)
    push!(qcbf, subroutine(9,qcme,(4:9) ))
    push!(qcbf2, control(9, (7,8),1=>X))
    push!(qcbf2, control(9, (7,9),2=>X)) 
    push!(qcbf2, control(9, (8,9),3=>X))  
    push!(qcbf, qcbf2)
    return qcbf
end
qcbf = bit_flip_rep_code()
CircuitStyles.barrier_for_chain[], temp = true, CircuitStyles.barrier_for_chain[]
draw = vizcircuit(qcbf; filename = "qcbf.pdf")
CircuitStyles.barrier_for_chain[] = temp

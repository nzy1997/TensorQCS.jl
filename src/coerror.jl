function coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real; cache::Union{Vector, Nothing} = nothing) where T
    appI = randn(T,size(u))*error_rate + I
    q2 , _ = qr(appI)
    q = u * q2
    cache === nothing || push!(cache, 1 - abs(tr(q'*u)/size(u,1)))
    return Matrix(q)
end

toput(gate::PutBlock) = gate
toput(gate::ControlBlock{XGate,1,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>ConstGate.CNOT)
toput(gate::ControlBlock{ZGate,1,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>ConstGate.CZ)

toput(gate::ControlBlock{XGate,2,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>CCX)
toput(gate::ControlBlock{ZGate,2,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>CCZ)
toput(gate::AbstractBlock) = gate

function error_quantum_circuit(qc::ChainBlock, error_rate::T ) where {T <: Real}
    pairs,vec = error_pairs(error_rate) 
    qcf = error_quantum_circuit(qc,pairs)
    return qcf, vec
end

function error_quantum_circuit(qc::ChainBlock, pairs)
    qcf = replace_block(x->toput(x), qc)
    for pa in pairs
        qcf = replace_block(pa, qcf)
    end
    return qcf
end

function error_pairs(error_rate::T) where {T <: Real}
    vec = Vector{T}()
    pairs = [x => matblock(coherent_error_unitary(mat(x),error_rate;cache =vec)) for x in [X,Z,H,CCZ,CCX,ConstGate.CNOT,ConstGate.CZ]]
    return pairs, vec
end

function add_indentity(qc::ChainBlock, qubits::Vector{Int})
	qc = simplify(qc; rules = [to_basictypes, Optimise.eliminate_nested])
    nq = nqubits(qc)
	qcn = chain(nq)
	idrs = []
    for i in 1:nq
        if i ∈ qubits
            idr = IdentityRecorder()
            push!(idrs,idr)
            push!(qcn, put(nq, i => idr))
        end
    end
	for gate in qc
		push!(qcn, gate)
        if !(toput(gate).content isa TrivialGate)
            for i in toput(gate).locs
                if i ∈ qubits
                    idr = IdentityRecorder()
                    push!(idrs,idr)
                    push!(qcn, put(nq, i => idr))
                end
            end
        end
	end
	return qcn,idrs
end


function error_location(qc::ChainBlock, qubits::Vector{Int},data_qubit::Int;nbatch = 1)
	qc = simplify(qc; rules = [to_basictypes, Optimise.eliminate_nested])
    nq = nqubits(qc)
	qcn = chain(nq)
    for i in 1:nq
        if i ∈ qubits
            qc1 = chain(nq)
            blk = test_position(qc1,qc,i,data_qubit;nbatch)
            blk === nothing || push!(qcn,put(nq,i =>blk))
        end
    end
    @show length(qc)
	for j in 1:length(qc)
        @show j
		push!(qcn, qc[j])
        if qc[j] isa MeasureAndReset 
            for i in toput(qc[j]).locations
                if i ∈ qubits
                    blk = test_position(qc[1:j],qc[j+1:end],i,data_qubit;nbatch)
                    blk === nothing || push!(qcn,put(nq,i =>blk))
                end
            end 
        elseif !(toput(qc[j]).content isa TrivialGate)
            for i in toput(qc[j]).locs
                if i ∈ qubits
                    blk = test_position(qc[1:j],qc[j+1:end],i,data_qubit;nbatch)
                    blk === nothing || push!(qcn,put(nq,i =>blk))
                end
            end
        end
	end

	return qcn
end

function is_error(qc::ChainBlock,data_qubit::Int;nbatch = 1)
    num_qubits = nqubits(qc)
    regrs = rand_state(1;nbatch)
    reg = join(zero_state(num_qubits - data_qubit;nbatch), regrs, zero_state(data_qubit-1;nbatch))
    reg0 = copy(reg)
    apply!(reg,qc)
   return !isapprox(fidelity(reg,reg0) , fill(1,nbatch))
end

function test_position(qc1::ChainBlock,qc2::ChainBlock,pos::Int,data_qubit::Int;nbatch = 1)
    elabel = ""
    qc = chain(qc1,put(nqubits(qc2),pos => Z),qc2)
    is_error(qc,data_qubit;nbatch) && (elabel = "Z" * elabel)

    qc = chain(qc1,put(nqubits(qc2),pos => Y),qc2)
    is_error(qc,data_qubit;nbatch) && (elabel = "Y" * elabel)

    qc = chain(qc1,put(nqubits(qc2),pos => X),qc2)
    is_error(qc,data_qubit;nbatch) && (elabel = "X" * elabel)

    if elabel == ""
        return nothing
    else 
        return addlabel(I2;color = length(elabel) == 3 ? "red" : "yellow",name=elabel)
    end
end
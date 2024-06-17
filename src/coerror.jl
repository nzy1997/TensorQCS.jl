function coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real; cache::Union{Vector, Nothing} = nothing) where T
    error_rate0 = error_rate
    for i in 1:100
        mtr ,inf = _coherent_error_unitary(u,error_rate0)
        if (inf/error_rate)<0.3
            error_rate0 = error_rate0 * 1.5
            continue
        elseif (inf/error_rate)>3
            error_rate0 = error_rate0 * 1.5
            continue
        end
        @show "Iteration!",i
        cache === nothing || push!(cache, inf)
        return mtr
    end
    mtr ,inf = _coherent_error_unitary(u,error_rate0)
    cache === nothing || push!(cache, inf)
    return mtr
end

function _coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real) where T
    appI = randn(T,size(u))*error_rate + I
    q2 , _ = qr(appI)
    q = u * q2
    return Matrix(q),1 - abs(tr(q'*u)/size(u,1))
end

@const_gate CCZ::ComplexF64 = diagm([1, 1,1,1,1,1,1,-1])
@const_gate CCX::ComplexF64 =  [1  0  0  0  0  0  0  0;
                                0  1  0  0  0  0  0  0;
                                0  0  1  0  0  0  0  0;
                                0  0  0  1  0  0  0  0;
                                0  0  0  0  1  0  0  0;
                                0  0  0  0  0  1  0  0;
                                0  0  0  0  0  0  0  1;
                                0  0  0  0  0  0  1  0]

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
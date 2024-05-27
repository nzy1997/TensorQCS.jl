struct ComplexConj{BT<:AbstractBlock,D} <: TagBlock{BT,D}
    content::BT
end
ComplexConj(x::BT) where {D,BT<:AbstractBlock{D}} = ComplexConj{BT,D}(x)
Yao.mat(::Type{T}, blk::ComplexConj) where {T} = conj(mat(T, content(blk)))

Base.conj(x::AbstractBlock) = ComplexConj(x)
Base.conj(x::ComplexConj) = content(x)
Base.copy(x::ComplexConj) = ComplexConj(copy(content(x)))

Base.conj(blk::ChainBlock{D}) where {D} =
    ChainBlock(blk.n, AbstractBlock{D}[conj(b) for b in reverse(subblocks(blk))])
Base.conj(x::PutBlock) = PutBlock(nqudits(x), conj(content(x)), x.locs)

function YaoBlocks.map_address(blk::ComplexConj, info::AddressInfo)
    ComplexConj(YaoBlocks.map_address(content(blk), info))
end
YaoBlocks.Optimise.to_basictypes(block::ComplexConj) = ComplexConj(block.content)

mutable struct SymbolRecorder{D} <: TrivialGate{D}
    symbol
end

SymbolRecorder(; nlevel=2) = SymbolRecorder{nlevel}(nothing)
Yao.nqudits(sr::SymbolRecorder) = 1
Yao.print_block(io::IO, sr::SymbolRecorder) = print(io, sr.symbol)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::SymbolRecorder, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "$(p.symbol)")])
end

function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,SymbolRecorder{D}}) where {T,D,C}
    lj = eb.slots[b.locs[1]]
    b.content.symbol = lj
    return eb
end

function dm_circ!(qcf::ChainBlock, qc::ChainBlock)
    num_qubits = nqubits(qc)
    @assert 2 * num_qubits == nqubits(qcf)
    push!(qcf,subroutine(2*num_qubits, qc, 1:num_qubits))
    push!(qcf,subroutine(2*num_qubits, conj(qc), num_qubits+1:2*num_qubits))
    return qcf
end

function ein_circ(qc::ChainBlock, input_qubits::Vector{Int}, output_qubits::Vector{Int})
    num_qubits = nqubits(qc)
    qc_f = chain(2*num_qubits)
    srs = [SymbolRecorder() for _ in 1:2*(length(input_qubits)+length(output_qubits))]
    srs_num = 1
    for i in input_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end

    dm_circ!(qc_f, qc)

    for i in output_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end
    return qc_f,srs
end

function old_ein_circ(qc::ChainBlock, data_qubits::Vector{Int}, num_qubits::Int)
    data_qubit_num = length(data_qubits)
    qc_f = chain(2*num_qubits)
    srs = [SymbolRecorder() for i in 1:(2*data_qubit_num+2*num_qubits)]
    srs_num = 1
    for i in data_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end

    for i in data_qubits
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end

    dm_circ!(qc_f, qc)

    for i in data_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end

    for i in data_qubits
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end
    for i in setdiff(1:num_qubits, data_qubits)
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end
    for i in setdiff(1:num_qubits, data_qubits)
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end
    return qc_f,srs
end

mapr(a::SymbolRecorder, b::SymbolRecorder) = a.symbol => b.symbol
function qc2enisum(qc::ChainBlock,srs::Vector{SymbolRecorder{D}},data_qubits::Vector{Int},num_qubits::Int) where D
    ein_code = yao2einsum(qc;initial_state=Dict(x=>0 for x in setdiff(setdiff(1:2*num_qubits, data_qubits), num_qubits .+ data_qubits)), optimizer=nothing)
    data_qubit_num = length(data_qubits)

    ds1 = 1:data_qubit_num
    ds2 = 2*data_qubit_num+1:3*data_qubit_num
    ds3 = data_qubit_num+1:2*data_qubit_num
    ds4 = 3*data_qubit_num+1:4*data_qubit_num
    anc1 =  4*data_qubit_num+1:num_qubits-data_qubit_num+4*data_qubit_num 
    anc2 = num_qubits+3*data_qubit_num+1:2*num_qubits+2*data_qubit_num
    jointcode = replace(ein_code.code, 
        mapr.(srs[ds1], srs[ds2])..., 
        mapr.(srs[ds3], srs[ds4])..., 
        mapr.(srs[anc1], srs[anc2])...)
    empty!(jointcode.iy) 
    return TensorNetwork(jointcode, ein_code.tensors)
end
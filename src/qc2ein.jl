struct ComplexConj{BT<:AbstractBlock,D} <: TagBlock{BT,D}
    content::BT
end
ComplexConj(x::BT) where {D,BT<:AbstractBlock{D}} = ComplexConj{BT,D}(x)
Yao.mat(::Type{T}, blk::ComplexConj) where {T} = conj(mat(T, content(blk)))

Base.conj(x::AbstractBlock) = ComplexConj(x)
Base.conj(x::ComplexConj) = content(x)
Base.copy(x::ComplexConj) = ComplexConj(copy(content(x)))

Base.conj(blk::ChainBlock{D}) where {D} =
    ChainBlock(blk.n, AbstractBlock{D}[conj(b) for b in subblocks(blk)])
Base.conj(x::PutBlock) = PutBlock(nqudits(x), conj(content(x)), x.locs)

Base.conj(blk::ControlBlock) =
    ControlBlock(blk.n, blk.ctrl_locs, blk.ctrl_config, conj(blk.content), blk.locs)

function YaoBlocks.map_address(blk::ComplexConj, info::AddressInfo)
    ComplexConj(YaoBlocks.map_address(content(blk), info))
end
YaoBlocks.Optimise.to_basictypes(block::ComplexConj) = ComplexConj(block.content)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::ComplexConj{<:PrimitiveBlock}, address, controls)
    bts = length(controls)>=1 ? YaoPlots.get_cbrush_texts(c, content(p)) : YaoPlots.get_brush_texts(c, content(p))
    YaoPlots._draw!(c, [controls..., (getindex.(Ref(address), occupied_locs(p)), bts[1], "conj of "*bts[2])])
end


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

struct ConnectMap 
    tr_qubits::Vector{Int}
    ptr_qubits::Vector{Int}
    nq::Int
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
    [push!(qc_f, put(2*num_qubits, input_qubits[i] => srs[2*i-1]), put(2*num_qubits, num_qubits+input_qubits[i] => srs[2*i])) for i in 1:length(input_qubits)]

    dm_circ!(qc_f, qc)

    [push!(qc_f, put(2*num_qubits, output_qubits[i] => srs[2*i-1+2*length(input_qubits)]), put(2*num_qubits, num_qubits+output_qubits[i] => srs[2*i+2*length(input_qubits)])) for i in 1:length(output_qubits)]
    return qc_f,srs
end

function ein_circ(qc::ChainBlock, cm::ConnectMap)
    return ein_circ(qc, cm.tr_qubits, cm.tr_qubits ∪ cm.ptr_qubits)
end

mapr(a::SymbolRecorder, b::SymbolRecorder) = a.symbol => b.symbol
function qc2enisum(qc::ChainBlock,zero_qubits::Vector{Int}, replace_dict::Dict{Int, Int})
    ein_code = yao2einsum(qc;initial_state=Dict(x=>0 for x in zero_qubits), optimizer=nothing)
    jointcode = replace(ein_code.code, replace_dict)
    empty!(jointcode.iy) 
    return TensorNetwork(jointcode, ein_code.tensors)
end

function qc2enisum(qc::ChainBlock, srs::Vector{SymbolRecorder{D}}, cm::ConnectMap) where D
    ein_code = yao2einsum(qc;initial_state=Dict(x=>0 for x in cm.ptr_qubits ∪ (cm.ptr_qubits.+cm.nq)), optimizer=nothing)
    replace_dict = ([[srs[2*i-1].symbol => srs[2*length(cm.tr_qubits)+2*i-1].symbol  for i in 1:length(cm.tr_qubits)]...,[srs[2*i].symbol => srs[2*length(cm.tr_qubits)+2*i].symbol  for i in 1:length(cm.tr_qubits)]...,[srs[4*length(cm.tr_qubits)+2*i-1].symbol => srs[4*length(cm.tr_qubits)+2*i].symbol for i in 1:length(cm.ptr_qubits)]...])
    jointcode = replace(ein_code.code, replace_dict...)
    empty!(jointcode.iy) 
    return TensorNetwork(jointcode, ein_code.tensors)
end

using Test
using TensorQCS
using TensorQCS.Yao
using TensorQCS.OMEinsum
using TensorQCS.LinearAlgebra
using Random

@testset "ComplexConj" begin
    ccj = ComplexConj(ConstGate.X)
    @test ccj isa ComplexConj
    @test apply!(product_state(bit"1"), ccj) ≈ apply!(product_state(bit"1"), ConstGate.X)
    @test conj(ccj) == ConstGate.X
end

@testset "SymbolRecorder" begin
    reg = rand_state(1)
    sr = SymbolRecorder()
    @test apply!(reg, sr) == reg

    srs = [SymbolRecorder() for i in 1:3]
    qc = chain(3,put(3, 1 => srs[1]), put(3, 2 => X), put(3, 3 => Y), put(3, 1 => Z), put(3, 2 => srs[2]), put(3, 3 => srs[3]))
    ein_code = yao2einsum(qc)
    @test srs[1].symbol == 1
    @test srs[2].symbol == 4
    @test srs[3].symbol == 5
end

@testset "ein_circ" begin
    u1 = mat(X)
    toyqc = chain(2, put(2,1 => GeneralMatrixBlock(u1; nlevel=2, tag="X")))
    qcf, srs = ein_circ(toyqc, ConnectMap([1],[2],2))

    @test length(srs) == 6
    ein_code = yao2einsum(qcf;optimizer=nothing)
end

function get_kraus(u::AbstractMatrix{T}, ndata::Int) where T
    return [u[i*2^ndata+1:(i+1)*2^ndata,1:2^ndata] for i in 0:2^(Yao.log2i(size(u, 1))-ndata)-1]
end

@testset "qc2enisum" begin
    Random.seed!(214)
    u1 = rand_unitary(2)
    u2 = rand_unitary(4)
    toyqc = chain(2, put(2,1 => GeneralMatrixBlock(u1; nlevel=2, tag="X")),put(2, (1,2) => GeneralMatrixBlock(u2; nlevel=2, tag="XCNOT")))
    cm = ConnectMap([1],[2],2)
    qcf, srs = ein_circ(toyqc, cm)
    tn = qc2enisum(qcf, srs, cm)
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    @show contract(optnet)[1]

    u = kron(u1,u1')
    kraus = get_kraus(u2, 1)
    uapp = mapreduce(x -> kron(x,x'), +, kraus)
    @test real(tr(u * uapp)) == real(contract(optnet)[1])
end



@testset "get_kraus" begin
    Random.seed!(214)
    u = rand_unitary(4)
    ub = matblock(u)
    reg1 = rand_state(1) 
    psi = join(zero_state(1),reg1)
    psi_f = apply(psi,ub)
    rho2 = partial_tr(density_matrix(psi_f),2)
    @test partial_tr(density_matrix(psi),2) == density_matrix(reg1)

    kraus = get_kraus(u,1)
    rho_f = zeros(ComplexF64,2,2)
    for idx in 1:2
       rho_f += kraus[idx] * partial_tr(density_matrix(psi),2).state * kraus[idx]' 
    end
    @test rho_f ≈ rho2.state

    u = rand_unitary(16)
    ub = matblock(u)
    reg1 = rand_state(2) 
    psi = join(zero_state(2),reg1)
    psi_f = apply(psi,ub)
    rho2 = partial_tr(density_matrix(psi_f),[3,4])
    @test partial_tr(density_matrix(psi),[3,4]) == density_matrix(reg1)

    kraus = get_kraus(u, 2)
    rho_f = zeros(ComplexF64,4,4)
    for idx in 1:4
       rho_f += kraus[idx] * partial_tr(density_matrix(psi), [3,4]).state * kraus[idx]' 
    end
    @test rho_f ≈ rho2.state
end
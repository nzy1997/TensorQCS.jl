using Test
using TensorQCS
using TensorQCS.Yao

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
    # u2 = mat(kron(I2,X)) * mat(cnot(2,2,1))
    toyqc = chain(2, put(2,1=>GeneralMatrixBlock(u1; nlevel=2, tag="X"))) #,put(2, (1,2) => GeneralMatrixBlock(u2; nlevel=2, tag="XCNOT")))
    qcf,srs = ein_circ(toyqc,[1],[2,1])
    @test length(srs) == 6

    ein_code = yao2einsum(qcf;optimizer=nothing)
end

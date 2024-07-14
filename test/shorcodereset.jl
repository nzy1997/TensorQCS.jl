using TensorQCS
using Test
using TensorQCS.TensorQEC.Yao

@testset "classical_decode" begin 
    @test classical_decode(bit"011011011")
    @test !classical_decode(bit"111000011")
    @test !classical_decode(bit"000011011")
    @test !classical_decode(bit"010010001")
    @test classical_decode(bit"111111111")
end

@testset "error_probabillity" begin
    @test error_probabillity(product_state(bit"000000011")) ≈ 1.0
    @test error_probabillity(product_state(bit"110000000")) ≈ 1.0
    @test error_probabillity(product_state(bit"000000001")) ≈ 0.0
    @test error_probabillity(product_state(bit"101010110000000001")) ≈ 0.0
end

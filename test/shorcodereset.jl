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
    onevec = [classical_decode(DitStr{2,9}(i)) for i in 0:511]
    @test error_probabillity(product_state(bit"000000011")) ≈ 1.0
    @test error_probabillity(product_state(bit"110000000")) ≈ 1.0
    @test error_probabillity(product_state(bit"000000001")) ≈ 0.0
    @test error_probabillity(product_state(bit"101010110000000001")) ≈ 0.0
    @test error_probabillity(product_state(bit"000000000111111111")) ≈ 1.0
    reg1 = product_state(bit"000000000111111111")
    reg2 = product_state(bit"000000000000000111") 
    @show error_probabillity((reg1+reg2)/sqrt(2))
    @show error_probabillity(product_state(bit"000000000")) 
end

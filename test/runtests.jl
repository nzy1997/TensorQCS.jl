using TensorQCS
using Test

@testset "qc2ein.jl" begin
   include("qc2ein.jl") 
end

@testset "coerror.jl" begin
   include("coerror.jl")
end
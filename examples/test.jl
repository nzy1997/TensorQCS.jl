using Test
using TensorQCS
using TensorQCS.Yao
@testset "reset_shor_circuit" begin
	qc, qcen = reset_shor_circuit()
	regrs = rand_state(1)
	reg = join(zero_state(12), regrs, zero_state(8))
	apply!(reg, subroutine(qcen, 1:9))
	apply!(reg, qc)
	apply!(reg, subroutine(qcen', 1:9))
	@test fidelity(reg, join(zero_state(12), regrs, zero_state(8))) ≈ 1
end

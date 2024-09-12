
function meandcr!(qc::ChainBlock, i::Int, st_me, qccr, num_qubits;translation = 0)
	qcme = chain(12)
	TensorQEC.measure_circuit!(qcme, st_me[i], 9 + mod1(i, 3))
	push!(qc, subroutine(qcme, (translation+1):(translation+12)))
	if mod(i, 3) == 0
		push!(qc, qccr[i-2])
		push!(qc, qccr[i-1])
		push!(qc, qccr[i])
		push!(qc, Measure(num_qubits; locs = (translation+10):(translation+12), resetto = bit"000"))
	end
end

function reset_shor_circuit()
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	st_me = stabilizers(ShorCode(), linearly_independent = false)
	num_qubits = 21

	st_pos = [19, 20, 21]
	qccr = chain(
		num_qubits,
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => Z),
		control(num_qubits, (st_pos[1], st_pos[3]), 4 => Z),
		control(num_qubits, (st_pos[2], st_pos[3]), 7 => Z),
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 2 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 3 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 4 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 5 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 6 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 7 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 8 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 9 => X),
	)
	# Z error, X stabilizers
	qc1 = chain(num_qubits)

	[push!(qc1, control(num_qubits,  9+i, i => X)) for i in 1:9]
	for i in 1:3
		meandcr!(qc1, i, st_me, qccr, num_qubits)
	end
	push!(qc1, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))

	# X error, Z stabilizers
	qc2 = chain(num_qubits)
	[push!(qc2, control(num_qubits, i, 9+i => X)) for i in 1:9]
	for i in 4:12
		meandcr!(qc2, i, st_me, qccr, num_qubits)
	end
	push!(qc2, Measure(num_qubits; locs = 10:18, resetto = bit"000000000"))
	qc3 = chain([put(num_qubits, i => X) for i in 1:9]...)
	return qc1,qc2,qc3,qcen
end


function reset_shor_circuit_nocopy()
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	st_me = stabilizers(ShorCode(), linearly_independent = false)
	num_qubits = 12

	st_pos = [10, 11, 12]
	qccr = chain(
		num_qubits,
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => Z),
		control(num_qubits, (st_pos[1], st_pos[3]), 4 => Z),
		control(num_qubits, (st_pos[2], st_pos[3]), 7 => Z),
		control(num_qubits, (st_pos[1], st_pos[2]), 1 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 2 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 3 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 4 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 5 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 6 => X),
		control(num_qubits, (st_pos[1], st_pos[2]), 7 => X),
		control(num_qubits, (st_pos[1], st_pos[3]), 8 => X),
		control(num_qubits, (st_pos[2], st_pos[3]), 9 => X),
	)
	# Z error, X stabilizers
	qc1 = chain(num_qubits)

	for i in 1:3
		meandcr!(qc1, i, st_me, qccr, num_qubits)
	end

	# X error, Z stabilizers
	qc2 = chain(num_qubits)
	for i in 4:12
		meandcr!(qc2, i, st_me, qccr, num_qubits)
	end

	qc3 = chain([put(num_qubits, i => X) for i in 1:9]...)
	return qc1,qc2,qc3,qcen
end

function error_circuit(error_rate)
	# pairs, vector = error_pairs(error_rate)
	pairs, vector = error_pairs(error_rate;gates=[X,Y,Z,H])
	# pairs, vector = error_pairs(error_rate;gates=[X,Y,Z,H,CCZ,ConstGate.Toffoli,ConstGate.CNOT,ConstGate.CZ])


	qc1,qc2,qcx,qcen = reset_shor_circuit_nocopy()
	num_qubits = nqubits(qc1)
	qc = chain(num_qubits)

	push!(qc, subroutine(qcen, 10:18))
	eqc1 = error_quantum_circuit(qc1, pairs)
	push!(qc, eqc1)

	push!(qc, put(21, 18 => H))
	push!(qc, subroutine(qcen, 10:18))
	eqc2 = error_quantum_circuit(qc2, pairs)
	push!(qc, eqc2)

	return qc, qcen, vector, error_quantum_circuit(chain(1,X), pairs),error_quantum_circuit(qcx,pairs)
	# return qc
end

function singleX(exqc;iters = 10)
	reg = zero_state(1)
	# reg = cu(reg)
	erp = Vector{Float64}()
	for i in 1:iters
		apply!(reg, exqc)
		apply!(reg, exqc)
		push!(erp,abs2(reg.state[2]))
		i%10 ==0 && print("i = $i ")
	end
	return erp
end

function TensorQEC.coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real;cache = nothing) where T
    error_rate0 = error_rate
    for i in 1:100
        u2,inf = _coherent_error_unitary2(u,error_rate0)
        if inf<error_rate/2
            error_rate0 = error_rate0*2
        elseif inf>error_rate*2
            error_rate0 = error_rate0/2
        else
            cache === nothing || push!(cache,inf)
            return u2
        end
    end
end
function _coherent_error_unitary2(u::AbstractMatrix{T}, error_rate::Real) where T
    appI = randn(T,size(u))*error_rate + I
    q2 , _ = qr(appI)
    q = u * q2
    return Matrix(q), 1 - abs(tr(q'*u)/size(u,1)) 
end

function reset_shor_code_total()
	qc1, qc2, qcx, qcen = reset_shor_circuit()
	qc = chain(21, subroutine(qcen, 10:18), qc1, put(21, 18 => H), subroutine(qcen, 10:18), qc2)
	qc = simplify(qc; rules = [to_basictypes, Optimise.eliminate_nested])
	return qc,qcen
end


function error_circuit_nocopy(error_rate)
	# pairs, vector = error_pairs(error_rate)
	pairs, vector = error_pairs(error_rate;gates=[X,Y,Z,H])
	# pairs, vector = error_pairs(error_rate;gates=[X,Y,Z,H,CCZ,ConstGate.Toffoli,ConstGate.CNOT,ConstGate.CZ])


	qc1,qc2,qcx,qcen = reset_shor_circuit_nocopy()
	num_qubits = nqubits(qc1)
	qc = chain(num_qubits)

	eqc1 = error_quantum_circuit(qc1, pairs)
	push!(qc, eqc1)


	eqc2 = error_quantum_circuit(qc2, pairs)
	push!(qc, eqc2)

	return qc, qcen, vector, error_quantum_circuit(chain(1,X), pairs),error_quantum_circuit(qcx,pairs)
	# return qc
end

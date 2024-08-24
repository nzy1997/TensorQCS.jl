function double_error_location(qc::ChainBlock, qubits::AbstractVector{Int}, skip_gates::AbstractVector{Int},add_error_qubits::AbstractVector{Int})
	nq = nqubits(qc)
	qcn = chain(nq)
	location_count = 0
    qc_locs = Int[]
    qubit_locs = Vector{Int}[]
	for i in 1:nq
		if i ∉ qubits
			location_count += 1
			push!(qcn, put(nq, i => addlabel(I2; color = "red", name = "$location_count 0 $i")))
            push!(qc_locs,0)
            push!(qubit_locs,[i])
		end
	end
	for j in 1:length(qc)
		@show j
		push!(qcn, qc[j])
		if j ∉ skip_gates
			if qc[j] isa MeasureAndReset
				for i in toput(qc[j]).locations
					location_count += 1
					push!(qcn, put(nq, i => addlabel(I2; color = "red", name = "$location_count $j $i")))
					push!(qc_locs,j)
					push!(qubit_locs,[i])
				end
			elseif !(toput(qc[j]).content isa TrivialGate)
				location_count += 1
				push!(qc_locs,j)
				push!(qubit_locs,collect(toput(qc[j]).locs))
				for i in toput(qc[j]).locs
					push!(qcn, put(nq, i => addlabel(I2; color = "red", name = "$location_count $j $i")))
				end
			end
		end
        if j ∈ add_error_qubits
            for i in qubits
                location_count += 1
                push!(qcn, put(nq, i => addlabel(I2; color = "red", name = "$location_count $j $i")))
                push!(qc_locs,j)
                push!(qubit_locs,[i])
            end 
        end
	end
	return qcn,qc_locs,qubit_locs
end


function is_error(qc::ChainBlock, data_qubit::Int)
	num_qubits = nqubits(qc)
	regrs = rand_state(1)
	reg = place_qubits(regrs,[data_qubit],num_qubits)
    reg = reg |> cu
	apply!(reg, qc)
	return !isapprox(fidelity(density_matrix(reg, data_qubit), density_matrix(regrs)), 1)
end


function check_qc(qcen::ChainBlock,qc::ChainBlock,qc_loc::AbstractVector,qubit_loc::AbstractVector,egate1,egate2)
   qcf = chain(21,subroutine(qcen,1:9),qc[1:qc_loc[1]])
   [push!(qcf, put(21,i=>egate1)) for i in qubit_loc[1]]
	push!(qcf, qc[(qc_loc[1]+1):qc_loc[2]])
	[push!(qcf, put(21,i=>egate2)) for i in qubit_loc[2]]
	push!(qcf, qc[(qc_loc[2]+1):end])
	push!(qcf,qc)
	push!(qcf,subroutine(qcen',1:9))
	return is_error(qcf, 9)
end

function check_qc3(qcen::ChainBlock,qc::ChainBlock,qc_loc::AbstractVector,qubit_loc::AbstractVector,egate1,egate2,egate3)
	qcf = chain(21,subroutine(qcen,1:9),qc[1:qc_loc[1]])
	[push!(qcf, put(21,i=>egate1)) for i in qubit_loc[1]]
	 push!(qcf, qc[(qc_loc[1]+1):qc_loc[2]])
	 [push!(qcf, put(21,i=>egate2)) for i in qubit_loc[2]]
	 push!(qcf, qc[(qc_loc[2]+1):qc_loc[3]])
	 [push!(qcf, put(21,i=>egate3)) for i in qubit_loc[3]]
	 push!(qcf, qc[(qc_loc[3]+1):end])
	 push!(qcf,qc)
	 push!(qcf,subroutine(qcen',1:9))
	 return is_error(qcf, 9)
 end

function check_double_pos(qcen::ChainBlock, qc::ChainBlock,qc_locs::AbstractVector,qubit_locs::AbstractVector)
	c = 0
	tc = 0
	all_combinations = combinations(1:length(qc_locs), 2)
	for comb in all_combinations
		for egate1 in [X,Y,Z]
			for egate2 in [X,Y,Z]
				tc += 1
				if check_qc(qcen,qc,qc_locs[comb],qubit_locs[comb],egate1,egate2) 
					c += 1
					@show qc_locs[comb],qubit_locs[comb],egate1,egate2
				end
			end
		end
		@show comb, c, tc
	end
	return c
end

function check_triple_pos(qcen::ChainBlock, qc::ChainBlock,qc_locs::AbstractVector,qubit_locs::AbstractVector)
	c = 0
	tc = 0
	all_combinations = combinations(1:length(qc_locs), 3)
	for comb in all_combinations
		for egate1 in [X,Y,Z]
			for egate2 in [X,Y,Z]
				for egate3 in [X,Y,Z]
					tc += 1
					if check_qc3(qcen,qc,qc_locs[comb],qubit_locs[comb],egate1,egate2,egate3)
						c += 1
						@show qc_locs[comb],qubit_locs[comb],egate1,egate2
					end
				end
			end
		end
		@show comb, c, tc
	end
	return c
end

function test_position(qc1::ChainBlock, qc2::ChainBlock, pos::Int, data_qubit::Int; nbatch = 1)
	elabel = ""
	qc = chain(qc1, put(nqubits(qc2), pos => Z), qc2)
	is_error(qc, data_qubit; nbatch) && (elabel = "Z" * elabel)

	qc = chain(qc1, put(nqubits(qc2), pos => Y), qc2)
	is_error(qc, data_qubit; nbatch) && (elabel = "Y" * elabel)

	qc = chain(qc1, put(nqubits(qc2), pos => X), qc2)
	is_error(qc, data_qubit; nbatch) && (elabel = "X" * elabel)

	if elabel == ""
		return nothing
	else
		return addlabel(I2; color = length(elabel) == 3 ? "red" : "yellow", name = elabel)
	end
end

function single_error_location(qc::ChainBlock, qubits::Vector{Int}, data_qubit::Int; nbatch = 1)
	qc = simplify(qc; rules = [to_basictypes, Optimise.eliminate_nested])
	nq = nqubits(qc)
	qcn = chain(nq)
	for i in 1:nq
		if i ∈ qubits
			qc1 = chain(nq)
			blk = test_position(qc1, qc, i, data_qubit; nbatch)
			blk === nothing || push!(qcn, put(nq, i => blk))
		end
	end
	@show length(qc)
	for j in 1:length(qc)
		@show j
		push!(qcn, qc[j])
		if qc[j] isa MeasureAndReset
			for i in toput(qc[j]).locations
				if i ∈ qubits
					blk = test_position(qc[1:j], qc[j+1:end], i, data_qubit; nbatch)
					blk === nothing || push!(qcn, put(nq, i => blk))
				end
			end
		elseif !(toput(qc[j]).content isa TrivialGate)
			for i in toput(qc[j]).locs
				if i ∈ qubits
					blk = test_position(qc[1:j], qc[j+1:end], i, data_qubit; nbatch)
					blk === nothing || push!(qcn, put(nq, i => blk))
				end
			end
		end
	end

	return qcn
end

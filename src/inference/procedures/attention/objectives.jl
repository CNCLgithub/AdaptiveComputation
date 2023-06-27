
export correspondence, td_flat, td_full

function n_obs(tr::Gen.Trace)
    @>> tr begin
        get_args
        first
        t -> tr[:kernel => t => :masks]
        length
    end
end

# Objectives

function td_assocs(tr::Gen.Trace)
    @>> tr begin
        get_args
        first
        t -> tr[:kernel => t]
        td_assocs
    end
end


"""
Computes the correspondence of a partition tensor.
"""
function correspondence(ptensor::BitArray{3}, ls::Vector{Float64})
    ls = ls .- logsumexp(ls)
    nx, ne, np = size(ptensor)
    c = zeros(nx, ne)
    @inbounds for p = 1:np
        c += ptensor[:, :, p] * exp(ls[p])
    end
    fill_p = 1.0 / nx
    # normalize each column to sum to 1
    @inbounds for e = 1:ne
        ce = @view c[:, e]
        sce = sum(ce)
        if sce == 0.
            ce[:] .= fill_p
        else
            ce ./= sce
        end
    end
    c
end

"""
Computes the correspondence of the resulting prediction RFS.
Defers to a state specific dispatch
"""
function correspondence(tr::Gen.Trace)
    @>> tr begin
        get_args
        first # time t
        t -> tr[:kernel => t] # state
        correspondence # state specific
    end
end

function td_flat(tr::Gen.Trace, temp::Float64)
    t = @>> tr begin
        get_args
        first # time t
    end
    # state specific flat
    td_flat(tr[:kernel => t], temp)
end

# target designation 1:
# scores and normalizes each partition SET
function td_full(pt::BitArray{3}, ls::Vector{Float64})
    nx, ne, np = size(pt)
    # no tracker elements
    ne === 0 && return Dict{BitVector, Float64}(falses(nx) => 0.)

    lne = log(ne)
    total = logsumexp(ls)
    td = Dict{BitVector, Float64}()
    @inbounds for p = 1:np
        # col vectors for which obs are included as sets of targets
        key = vec(reduce(|, pt[:, :, p], dims = 2))
        td[key] = haskey(td, key) ? logsumexp(td[key], ls[p]) : ls[p]
    end
    # map!(x -> x - total, values(td))
    td
end

function td_full(tr::Gen.Trace)
    @>> tr begin
        get_args
        first # time t
        t -> tr[:kernel => t] # state
        td_full # state specific obj
    end
end

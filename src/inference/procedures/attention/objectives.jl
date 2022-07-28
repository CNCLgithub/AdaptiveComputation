
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
function correspondence(ptensor::BitArray{3}, ls::Vector{Float64}) where {T}
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

# target designation 2:
# for each observation gets score for being a target
function td_flat(pt::BitArray{3}, ls::Vector{Float64})
    tw = ones(size(pt, 2))
    td_flat(pt, ls, tw)
end
function td_flat(pt::BitArray{3}, ls::Vector{Float64}, tw::Vector{Float64})
    nx, ne, np = size(pt)
    @assert ne === length(tw) "Number of elements must match target weights"
    @assert np === length(ls) "Partition count in `pt` must match log score"

    td = Dict{Int64, Float64}()
    if ne === 0
        for k = 1:nx
            td[k] = 0.0 # log(1.0)
        end
        return td
    end
    total_lse = logsumexp(ls)
    @inbounds for x = 1:nx
        tdx = Vector{Float64}(undef, np)
        # @show x
        for p = 1:np
            # @show (pt[x, :, p] .* tw, ls[p])
            tdx[p] = log(sum(pt[x, :, p] .* tw)) + ls[p]
        end
        # @show tdx
        td[x] = logsumexp(tdx) - total_lse
    end
    td
end


function td_flat(tr::Gen.Trace)
    @>> tr begin
        get_args
        first # time t
        t -> tr[:kernel => t] # state
        td_flat # state specific flat
    end
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

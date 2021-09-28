
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

function assocs(tr::Gen.Trace)
    @>> tr begin
        get_args
        first
        t -> tr[:kernel => t]
        assocs
    end
end

function correspondence(ptensor::BitArray{3}, ls::Vector{Float64}) where {T}
    ls = ls .- logsumexp(ls)
    nx, ne, np = size(ptensor)
    c = zeros(nx, ne)
    @inbounds for p = 1:np
        c += ptensor[:, :, p] * exp(ls[p])
    end
    return c
end

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
    nx, ne, np = size(pt)
    td = Dict{Int64, Float64}()
    if ne === 0
        for k = 1:nx
            td[k] = 0.0 # log(1.0)
        end
        return td
    end

    total_lse = logsumexp(ls)
    # @show ls
    # @show total_lse
    @inbounds for x = 1:nx
        assigned = vec(reduce(|, pt[x, :, :], dims = 1))
        td[x] = sum(assigned) === 0 ? -Inf : logsumexp(ls[assigned]) - total_lse
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

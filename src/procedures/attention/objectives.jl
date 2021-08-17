
# Objectives

# target designation 2:
# for each observation gets score for being a target
function _td2(xs::Vector{T}, pmbrfs::RFSElements{T}, t::Int) where {T}
    @assert first(pmbrfs) isa PoissonElement "First element assumed to be clutter"

    ls, cube = GenRFS.associations(pmbrfs, xs)
    # ls = ls .- logsumexp(ls)
    cube = cube[:, 2:end, :]
    nx, ne, np = size(cube)
    td = Dict{Int64, Float64}()
    # assuming first element is pmbrfs
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
        # assigned = @>> cube[x, : ,:] begin
        #     eachcol
        #     map(any)
        # end
        assigned = vec(reduce(|, cube[x, :, :], dims = 1))
        # @show assigned
        # @show logsumexp(ls[assigned])
        td[x] = sum(assigned) === 0 ? -Inf : logsumexp(ls[assigned]) # - total_lse
    end
    td
end

# target designation 1:
# scores and normalizes each partition SET
function _td(xs::Vector{T}, pmbrfs::RFSElements{T}, t::Int) where {T}
    @assert first(pmbrfs) isa PoissonElement "First element assumed to be clutter"

    ls, cube = GenRFS.associations(pmbrfs, xs)
    # ls = ls .- logsumexp(ls)
    cube = cube[:, 2:end, :]
    nx, ne, np = size(cube)
    td = Dict{BitVector, Float64}()
    # assuming first element is pmbrfs
    if ne === 0
        key = falses(nx)
        td[key] = 0.0 # log(1.0)
        return td
    end

    @inbounds for p = 1:np
        # col vectors for which obs are included as sets of targets
        # TODO: see if there is a way to easily do this col-wise
        key = vec(reduce(|, cube[:, :, p], dims = 2))
        if haskey(td, key)
            td[key] = logsumexp(td[key], ls[p])
        else
            td[key] = ls[p]
        end
    end
    td
end

# returns a vector of target designation distributions
# for each receptive_field
function target_designation_receptive_fields(tr::Gen.Trace)
    t = first(Gen.get_args(tr))

    rfs_vec = @>> Gen.get_retval(tr) begin
        last # get the states
        last # get the last state
        (cg -> get_prop(cg, :rfs_vec)) # get the receptive fields
    end # rfes for each rf

    receptive_fields = @> tr begin
        get_choices
        get_submap(:kernel => t => :receptive_fields)
        get_submaps_shallow
        # vec of tuples (rf id, rf mask choicemap)
    end # masks for each rf

    # @debug "receptive fields $(typeof(receptive_fields[1]))"
    tds = @>> receptive_fields begin
        map(rf -> _td(convert(Vector{BitMatrix}, rf[2][:masks]), rfs_vec[rf[1]], t))
        # map(rf -> _td2(convert(Vector{BitMatrix}, rf[2][:masks]), rfs_vec[rf[1]], t))
    end
end

# returns target designation distribution
function target_designation(tr::Gen.Trace)
    t = first(Gen.get_args(tr))
    xs = get_choices(tr)[:kernel => t => :masks]
    pmbrfs = Gen.get_retval(tr)[2][t].rfs

    current_td = _td(xs, pmbrfs, t)
end

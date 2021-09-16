export Diff

struct Diff
    # persistence
    born::Vector{Thing}
    died::Vector{Int64}
    # mutability
    static::Vector{Int64}
    changed::Dict{Int64, Thing}
end

function Diff(cg::CausalGraph, born::Vector{Thing},
              died::Vector{Int64})
    Diff(born, died, Int64[], Dict{Int64, Thing}())
end


function Diff(lf::Diff, cg::CausalGraph, chng_idxs::Vector{Int64},
              changed::AbsactVector{Thing})

    @unpack born, died = lf

    dm = get_dm(cg)
    st = static(dm, cg)

    chng_d = Dict{Int64, Thing}()
    @inbounds for i = 1:length(chng_idxs)
        chng_d[chng_idxs[i]] = changed[i]
    end

    Diff(born, died, static, changed)
end

function static(dm::InertiaModel, cg::CausalGraph)
    get_object_verts(cg, Wall)
end

function death(dm::InertiaModel, cg::CausalGraph)
    gm = get_gm(cg)
    @unpack death_rate = gm
    vs = get_object_verts(cg, Dot)
    n = length(vs)
    elements = RFSElements{Int64}(undef, n)
    n === 0 && return elements
    @inbounds for i = 1:n
        elements[i] = BernoulliElement{Int64}(death_rate,
                                              id_dist, (vs[i],))
    end
    elements
end


function birth_diff(dm::InertiaModel, cg::CausalGraph,
                    born::AbstractArray{Thing},
                    died::AbstractArray{Int64})
    ens_idx = @> cg get_object_verts(UniformEnsemble) first
    ens = UniformEnsemble(cg, died, born)
    changed = Dict{Int64, Thing}(ens_idx => ens)
    Diff(born, died, static(dm, cg), changed)
end

function birth_limit(dm::InertiaModel, cg::CausalGraph)
    gm = get_gm(cg)
    nthings = get_object_verts(cg, Dot)
    gm.max_things - nthings
end

function birth_args(dm::InertiaModel, cg::CausalGraph, n::Int64)
    gm = get_gm(cg)
    fill(gm, n)
end


walls_idx(dm::InertiaModel) = collect(1:4)

function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

function inertia_step_args(cg::CausalGraph)
    vs = get_object_verts(cg, Dot)
    cgs = fill(cg, length(vs))
    (cgs, vs)
end

function cross2d(a::Vector{Float64}, b::Vector{Float64})
    a1, a2 = a
    b1, b2 = b
    a1*b2-a2*b1
end

function vector_to(w::Wall, o::Object)
    # w.n/norm(w.n) .* dot(o.pos[1:2] - w.p1, w.n)
    # from https://stackoverflow.com/a/48137604
    @unpack p1, p2, n = w
    p3 = o.pos[1:2]
    -n .* cross2d(p2-p1,p3-p1) ./ norm(p2-p1)
end

function vector_to(a::Object, b::Object)
    b.pos[1:2] - a.pos[1:2]
end

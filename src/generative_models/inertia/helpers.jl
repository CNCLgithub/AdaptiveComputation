struct InertiaKernelState
    world::CausalGraph
    es::RFSElements
    xs::AbstractArray
    pt::BitArray{3}
    pls::Vector{Float64}
end

function InertiaKernelState(world::CausalGraph,
                            es::RFSElements{T},
                            xs::AbstractArray{T}) where {T}
    (pls, pt) = GenRFS.associations(es, xs)
    InertiaKernelState(world, es, xs, pt, pls)
end

function assocs(st::InertiaKernelState)
    (st.pt, st.pls)
end


function correspondence(st::InertiaKernelState)
    @unpack es, pt, pls = st
    # all trackers (anything that isnt an ensemble)
    tracker_ids = findall(x -> !isa(x, UniformEnsemble), es)
    pt = pt[:, tracker_ids, :]
    correspondence(pt, pls)
end

function td_flat(st::InertiaKernelState)
    @unpack es, pt, pls = st
    # all trackers (anything that isnt an ensemble)
    tracker_ids = findall(x -> !isa(x, UniformEnsemble), es)
    pt = pt[:, tracker_ids, :]
    td_flat(pt, pls) # P(x_i = Target)
end

function td_full(st::InertiaKernelState)
    @unpack es, pt, pls = st
    # all trackers (anything that isnt an ensemble)
    tracker_ids = findall(x -> !isa(x, UniformEnsemble), es)
    pt = pt[:, tracker_ids, :]
    td_full(pt, pls) # P({x...} are targets)
end

function trackers(dm::InertiaModel, tr::Trace)
    t = first(get_args(tr))
    changed = tr[:kernel => t => :dynamcis => :trackers]
    n_chng = length(changed)
    n_born = tr[:kernel => t => :epistemics => :to_birth]
    ts = Vector{Pair}(undef, n_chng + n_born)
    # while the vertices of trackers may not be contiguous
    # across time steps, trackers order is preserved
    # First updated trackers (`changed`) and then new trackers.
    @inbounds for i = 1:n_chng
        ts[i] = :kernel => t => :dynamics => :trackers => i
    end
    @inbounds for i = 1:n_born
        ts[i + n_chng] = :kernel => t => :epistemics => :birth => i
    end
    ts
end

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
    st = StaticPath[]
    for w in get_object_verts(cg, Wall)
        push!(st, w => :object)
    end
    ens_idx = @> cg get_object_verts(UniformEnsemble) first
    prev_ens = get_prop(cg, ens_idx, :object)
    ens = UniformEnsemble(cg, died, born, prev_ens)
    changed = Dict{ChangeDiff, Thing}((ens_idx => :object) => ens)
    Diff(born, died, st, changed)
end

function birth_limit(dm::InertiaModel, cg::CausalGraph)
    gm = get_gm(cg)
    nthings = length(get_object_verts(cg, Dot))
    gm.max_things - nthings !== 0 # 1 or 0
end

function birth_args(dm::InertiaModel, cg::CausalGraph, n::Int64)
    gm = get_gm(cg)
    (fill(gm, n), fill(dm, n))
end


"""
Creates a `Diff` with `:object` updates for each tracker.
Also propagates graphics state
"""
function diff_from_trackers(vs::Vector{Int64}, trackers::AbstractArray{<:Thing})
    chng = ChangeDict()
    st = Vector{StaticPath}(undef, length(vs))
    @inbounds for i = 1:length(vs)
        chng[vs[i] => :object] = trackers[i]
        st[i] = vs[i] => :flow
    end
    Diff(Thing[], Int64[], st, chng)
end


walls_idx(dm::InertiaModel) = collect(1:4)

function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

function inertia_step_args(cg::CausalGraph, d::Diff)
    vs = @> cg get_object_verts(Dot) setdiff(d.died)
    # vs = get_object_verts(cg, Dot)
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

function UniformEnsemble(gm::GMParams, gr::Graphics, rate,
                         targets)
    n_receptive_fields = length(gr.receptive_fields)
    rate_per_field = rate / n_receptive_fields

    r = ceil(gm.dot_radius * gr.img_dims[1] / gm.area_width)
    n_pixels_rf = @>> gr.receptive_fields first get_dimensions prod
    pixel_prob =  ((2 * pi * r^2) / n_pixels_rf) * rate_per_field

    UniformEnsemble(rate_per_field, pixel_prob, targets)
end


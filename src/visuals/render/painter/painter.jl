export Painter, paint, InitPainter

abstract type Painter end

# function paint_series(cgs::Vector{CausalGraph}, painters::Vector{Vector{Painter}})
#     for (cg, tp) in zip(cgs, painters)
#         foreach(p -> paint(p, cg), tp)
#         finish()
#     end
#     return nothing
# end

# function paint_series(cgs_vec::Vector{Vector{CausalGraph}},
#                       painters_vec::Vector{Vector{Vector{Painter}}})
#     for (cgs, painters_t) in zip(cgs_vec, painters_vec)
#         for (cg, painters) in zip(cgs, painters_t)
#             foreach(p -> paint(p, cg), painters)
#         end
#         finish()
#     end
#     return nothing
# end

# function paint(p::Painter, cg::CausalGraph)
#     @>> cg edges map(e -> paint(p, cg, e))
#     @>> cg vertices map(v -> paint(p, cg, v))
#     return nothing
# end

# function paint(p::Painter, cg::CausalGraph, v::Int64)
#     paint(p, cg, v, get_prop(cg, v, :object))
#     return nothing
# end

# function paint(p::Painter, cg::CausalGraph, v::Int64, o::Object)
#     return nothing
# end

@with_kw struct InitPainter <: Painter
    path::String
    dimensions::Tuple{Int64, Int64}
    background::String = "#e7e7e7"
end

function paint(p::InitPainter, st::GMState)
    error("not implemented")
end


include("psiturk.jl")
include("id.jl")
include("kinematics.jl")
include("attention_rings.jl")
# include("internal_force.jl")
# include("poly.jl")
# include("subset.jl")
# include("target.jl")
# include("poiss_dot.jl")
# include("rf.jl")
# include("attention_gaussian.jl")
include("attention_centroid.jl")
# include("fixations_painter.jl")

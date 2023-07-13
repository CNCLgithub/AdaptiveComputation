export Painter, paint, InitPainter

abstract type Painter end

function paint end

@with_kw struct InitPainter <: Painter
    path::String
    dimensions::Tuple{Int64, Int64}
    background::String = "#e7e7e7"
end


function paint(p::InitPainter, ::GMState)
    height, width = p.dimensions
    Drawing(width, height, p.path)
    Luxor.origin()
    background(p.background)
end

include("psiturk.jl")
include("id.jl")
include("kinematics.jl")
include("attention_rings.jl")
include("attention_centroid.jl")
# include("fixations_painter.jl")

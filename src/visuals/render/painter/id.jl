export IDPainter

@with_kw struct IDPainter <: Painter
    colors::Vector
    label::Bool = true
    label_size::Float64 = 40.0
    alpha::Float64 = 1.0
end

function remap(p::IDPainter, vmap::Vector{Int64})::IDPainter
    isempty(p.colors) && return p
    IDPainter(p.colors[sortperm(vmap)], p.label, p.label_size, p.alpha)
end


function paint(p::IDPainter, cg::CausalGraph, v::Int64, d::Dot)
    if !isempty(p.colors)
        c = p.colors[v]
        _draw_circle(d.pos[1:2], d.radius, c,
                     opacity = p.alpha)
    end
    p.label && _draw_text("$v", d.pos[1:2] .+ [d.width/2, d.height/2],
                          size = p.label_size)
    return nothing
end

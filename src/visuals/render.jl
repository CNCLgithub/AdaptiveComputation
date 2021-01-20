export render

using Luxor, ImageMagick

"""
initialize the drawing file according to the frame number,
position at (0,0) and set background color
"""
function _init_drawing(frame, path, gm;
                       background_color="ghostwhite")
    fname = "$(lpad(frame, 3, "0")).png"
    Drawing(gm.area_width, gm.area_height,
            joinpath(path, fname))
    origin()
    background(background_color)
end

"""
helper to draw text
"""
function _draw_text(text, position; opacity=1.0, color="black", size=30)
    setopacity(opacity)
    sethue(color)
    Luxor.fontsize(size)
    point = Luxor.Point(position[1], -position[2])
    Luxor.text(text, point, halign=:right, valign=:bottom)
end

"""
helper to draw circle
"""
function _draw_circle(position, radius, color;
                      opacity=1.0, style=:fill,
                      pattern="solid")
    if style==:stroke
        setline(5)
        setdash(pattern)
    end
    setopacity(opacity)
    sethue(color)
    point = Luxor.Point(position[1], -position[2])
    Luxor.circle(point, radius, style)
end

"""
helper to draw array (used to draw the predicted tracker masks)
"""
function _draw_array(array, gm, color; opacity=1.0)
    sethue(color)

    tiles = Tiler(gm.area_width, gm.area_height, gm.img_width, gm.img_height, margin=0)

    for (pos, n) in tiles
        # reading value from the array
        row = tiles.currentrow
        col = tiles.currentcol
        value = array[row, col]

        # scaling opacity according to the value
        setopacity(opacity*value)

        box(pos, tiles.tilewidth, tiles.tileheight, :fill)
    end
end

"""
helper to draw arrow
"""
function _draw_arrow(startpoint, endpoint, color;
                     opacity=1.0, style=:fill,
                     linewidth=5.0, arrowheadlength=15.0)
    setopacity(opacity)
    sethue(color)
    p1 = Luxor.Point(startpoint[1], -startpoint[2])
    p2 = Luxor.Point(endpoint[1], -endpoint[2])
    Luxor.arrow(p1, p2, linewidth=linewidth, arrowheadlength=arrowheadlength)
end

function render_object(object::Object)
    error("not defined")
end

function render_object(dot::Dot;
                       leading_edges=true,
                       dot_color="#e0b388",
                       leading_edge_color="#ee70f2")

    color = dot.probe ? probe_color : dot_color

    _draw_circle(dot.pos[1:2], dot.radius, color)
    if leading_edges
        _draw_circle(dot.pos[1:2], dot.radius, leading_edge_color, style=:stroke)
    end
end

"""
renders the causal graph
"""
function render_cg(cg::CausalGraph, gm::GMParams;
                   show_label=true,
                   highlighted::Vector{Int}=Int[],
                   highlighted_color="blue",
                   render_edges=false)

    objects = cg.elements

    # furthest (highest z) comes first in depth_perm
    depth_perm = sortperm(map(x -> x.pos[3], objects), rev=true)

    for i in depth_perm
        render_object(objects[i])
        if show_label && !isa(objects[i], Pylon)
            _draw_text("$i", objects[i].pos[1:2] .+ [objects[i].width/2, objects[i].height/2])
        end
        if i in highlighted
            _draw_circle(objects[i].pos[1:2], objects[i].width, highlighted_color;
                         style=:stroke, pattern="dash")
        end
    end
end




"""
renders detailed information about inference on top of stimuli

gm - generative model parameters

optional:
stimuli - true if we want to render without timestep and inference information
freeze_time - time before and after movement (for highlighting targets and querying)
highlighted - array 
"""
function render(gm, T;
                gt_causal_graphs=nothing,
                dot_positions=nothing,
                probes=nothing,
                stimuli=false,
                freeze_time=0,
                highlighted=Int[],
                background_color="#7079f2",
                path="render")

    # stopped at beginning
    for t=1:freeze_time
        _init_drawing(t, path, gm,
                      background_color = background_color)

        if !isnothing(gt_causal_graphs)
            render_cg(gt_causal_graphs[1], gm;
                      highlighted=highlighted,
                      show_label=!stimuli)
        end
        finish()
    end

    # tracking while dots are moving
    # TODO change with good init causal graphs
    for t=1:T
        print("render timestep: $t/$T \r")
        _init_drawing(t+freeze_time, path, gm,
                      background_color = background_color)

        if !stimuli
            _draw_text("$t", [gm.area_width/2 - 100, gm.area_height/2 - 100], size=50)
        end

        if !isnothing(gt_causal_graphs)
            render_cg(gt_causal_graphs[t+1], gm;
                      show_label=!stimuli)
        end
    end

    # final freeze showing the query
    for t=1:freeze_time
        _init_drawing(t+T+freeze_time, path, gm,
                      background_color = background_color)

        if !isnothing(gt_causal_graphs)
            render_cg(gt_causal_graphs[T+1], gm;
                      highlighted=highlighted,
                      highlighted_color="#d2f72a",
                      show_label=!stimuli)
        end

        finish()
    end
end

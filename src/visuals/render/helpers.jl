
"""
    initialize the drawing file according to the frame number,
    position at (0,0) and set background color
"""
function _init_drawing(frame, path, gm;
                       receptive_fields = nothing,
                       receptive_fields_overlap = 0)
    fname = "$(lpad(frame, 3, "0")).png"
    Drawing(round(Int, gm.area_width), round(Int, gm.area_height),
            joinpath(path, fname))
    origin()
    background("#e7e7e7")

    # drawing receptive_fields
    if !isnothing(receptive_fields)
        sethue("black")
        tiles = Tiler(gm.area_width, gm.area_height, receptive_fields[1], receptive_fields[2], margin=0)
        foreach(tile -> box(tile[1], tiles.tilewidth, tiles.tileheight, :stroke), tiles)
        setopacity(0.1)
        setline(receptive_fields_overlap/gm.img_width*gm.area_width*2)
        foreach(tile -> box(tile[1], tiles.tilewidth, tiles.tileheight, :stroke), tiles)
    end
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
                     opacity=1.0,
                     linewidth=5.0, arrowheadlength=15.0)
    setopacity(opacity)
    sethue(color)
    p1 = Luxor.Point(startpoint[1], -startpoint[2])
    p2 = Luxor.Point(endpoint[1], -endpoint[2])
    Luxor.arrow(p1, p2, linewidth=linewidth, arrowheadlength=arrowheadlength)
end

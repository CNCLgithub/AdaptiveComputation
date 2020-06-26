export render

using Luxor

function _init_drawing(frame, dir, gm;
                       background_color="ghostwhite")
    fname = "$(lpad(frame, 3, "0")).png"
    Drawing(gm.area_width, gm.area_height,
            joinpath(dir, fname))
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
function _draw_circle(position, radius, color; opacity=1.0, style=:fill)
    setopacity(opacity)
    sethue(color)
    point = Luxor.Point(position[1], -position[2])
    Luxor.circle(point, radius, style)
end

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

function _render_dots(dot_positions_t,
                      gm;
                      show_label=true, 
                      dot_color="lightsalmon2",
                      highlighted::Union{Nothing,Vector{Int}} = nothing,
                      highlighted_color = "red")
    
    for i=1:size(dot_positions_t, 1)
        _draw_circle(dot_positions_t[i,1:2], gm.dot_radius, dot_color)

        if show_label
            _draw_text("$i", dot_positions_t[i,1:2] .+ gm.dot_radius)
        end
    end

    if !isnothing(highlighted)
        for i in highlighted
            _draw_circle(dot_positions_t[i,1:2], gm.dot_radius/2.0, highlighted_color)
        end
    end
end

"""
    renders particle filter inferred positions of the tracked objects
"""
function _render_pf(pf_xy_t,
                    gm;
                    pf_color="darkslateblue",
                    attended=nothing,
                    tracker_masks=nothing,
                    tracker_masks_colors=["indigo", "green", "blue", "yellow"])
    
    n_particles, n_trackers, _ = size(pf_xy_t)

    for p=1:n_particles
        for i=1:n_trackers

            # drawing the predicted position of this tracker in particle
            pred_position = pf_xy_t[p,i,:]

            # if we don't have tracker masks, then draw the little predicted positions
            if isnothing(tracker_masks)
                _draw_circle(pred_position, gm.dot_radius/4, pf_color)
            end

            _draw_text("$i", pred_position)
            
            # visualizing attention
            if !isnothing(attended)
                bg_visibility = 1.0 - p*(1.0/n_particles)
                prev_bg_visibility = 1.0 - (p-1)*(1.0/n_particles)
                opacity = 1.0 - bg_visibility/prev_bg_visibility
                opacity *= attended[i]
                _draw_circle(pred_position, 2*gm.dot_radius, "red", opacity=opacity, style=:stroke)
            end

            if !isnothing(tracker_masks)
                bg_visibility = 1.0 - p*(1.0/n_particles)
                prev_bg_visibility = 1.0 - (p-1)*(1.0/n_particles)
                opacity = 1.0 - bg_visibility/prev_bg_visibility
                opacity *= 0.5
                _draw_array(tracker_masks[p,i], gm, tracker_masks_colors[i], opacity=opacity)
            end
        end
    end

end


"""
    renders detailed information about inference on top of stimuli
"""
function render(dot_positions,
                q,
                gm;
                stimuli=false,
                pf_xy=nothing,
                dir="render",
                freeze_time=0,
                highlighted=nothing,
                attended=nothing,
                tracker_masks=nothing)

    println("rendering inference info on stimuli...")
    mkpath(dir)

    # stopped at beginning
    for t=1:freeze_time
        _init_drawing(t, dir, gm)
        
        _render_dots(dot_positions[1], gm; highlighted=collect(1:gm.n_trackers))
    
        if !isnothing(pf_xy)
            _render_pf(pf_xy[1,:,:,:], gm, tracker_masks=tracker_masks[1,:,:])
        end

        finish()

    end
    
    # tracking while dots are moving
    for t=1:q.k
        println("timestep: $t")
        _init_drawing(t+freeze_time, dir, gm)
        _draw_text("$t", [gm.area_width/2 - 100, gm.area_height/2 - 100], size=50)

        _render_dots(dot_positions[t], gm)
    
        if !isnothing(pf_xy)
            _render_pf(pf_xy[t,:,:,:], gm; attended=attended[t], tracker_masks=tracker_masks[t,:,:])
        end

        finish()
    end
    
    # final freeze time showing the answer
    for t=1:freeze_time
        _init_drawing(t+q.k+freeze_time, dir, gm)

        _render_dots(dot_positions[q.k], gm; highlighted=collect(1:gm.n_trackers))
    
        if !isnothing(pf_xy)
            _render_pf(pf_xy[q.k,:,:,:], gm; attended=attended[q.k], tracker_masks=tracker_masks[q.k,:,:])
        end

        finish()
    end
end


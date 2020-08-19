export render

using Luxor, ImageMagick


"""
    initialize the drawing file according to the frame number,
    position at (0,0) and set background color
"""
function _init_drawing(frame, path, gm, prefix;
                       background_color="ghostwhite")
    fname = prefix * "$(lpad(frame, 3, "0")).png"
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
function _draw_circle(position, radius, color; opacity=1.0, style=:fill)
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

function _render_probes(dot_positions_t,
                        probes_t,
                        gm;
                        probe_color="coral1")
    for i=1:size(dot_positions_t, 1)
        if probes_t[i]
            _draw_circle(dot_positions_t[i,1:2], 0.5 * gm.dot_radius, probe_color, opacity = 0.8)
        end
    end
end

"""
    helper to draw the dots
"""
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
    helper function to determine opacity based on number of particles
"""

function _get_opacity(particle, n_particles)
    bg_visibility = 1.0 - particle*(1.0/n_particles)
    prev_bg_visibility = 1.0 - (particle-1)*(1.0/n_particles)
    opacity = 1.0 - bg_visibility/prev_bg_visibility
end

"""
    helper function to draw the tracker masks on top of the image
"""
function _render_tracker_masks(tracker_masks;
                               tracker_masks_colors=["indigo", "green", "blue", "yellow"])

    n_particles, n_trackers, _ = size(tracker_masks)

    for p=1:n_particles
        for i=1:n_trackers
            opacity = 0.5 * _get_opacity(p,n_particles)
            _draw_array(tracker_masks[p,i], gm, tracker_masks_colors[i], opacity=opacity)
        end
    end
end

"""
    renders particle filter inferred positions (and velocities) of the tracked objects
"""
function _render_pf(pf_xy_t,
                    gm;
                    pf_color="darkslateblue",
                    attended=nothing,
                    pf_vel=nothing,
                    show_label=true)
    
    n_particles, n_trackers, _ = size(pf_xy_t)

    for p=1:n_particles
        for i=1:n_trackers

            # drawing the predicted position of this tracker in particle
            pred_position = pf_xy_t[p,i,:]

            _draw_circle(pred_position, gm.dot_radius/3, pf_color, opacity=1.0)

            if !isnothing(pf_vel)
                _draw_arrow(pred_position, pred_position + pf_vel[p,i,:], pf_color,
                            opacity=1.0)
            end
            
            if show_label
                _draw_text("$i", pred_position)
            end
            
            # visualizing attention
            if !isnothing(attended)
                att_opacity = attended[i] * _get_opacity(p, n_particles)
                _draw_circle(pred_position, 2.5*gm.dot_radius, "red", opacity=att_opacity, style=:stroke)
            end
        end
    end

end


"""
    renders detailed information about inference on top of stimuli
"""
function render(gm;
                stimuli=false,
                array=false,
                dot_positions=nothing,
                pf_xy=nothing,
                pf_vel=nothing,
                path="render",
                prefix="",
                freeze_time=0,
                highlighted=nothing,
                attended=nothing,
                probes=nothing,
                tracker_masks=nothing)
    
    # if returning array of images as matrices, then make vector
    array ? imgs = [] : mkpath(path)
    
    # getting number of timesteps
    k = size(dot_positions, 1)

    # stopped at beginning
    for t=1:freeze_time
        _init_drawing(t, path, gm, prefix)
        
        if !isnothing(dot_positions)
            _render_dots(dot_positions[1], gm;
                         highlighted=collect(1:gm.n_trackers),
                         show_label=!stimuli)
        end
    
        if !isnothing(pf_xy)
            pf_vel_t = isnothing(pf_vel) ? nothing : pf_vel[1,:,:,:]
            _render_pf(pf_xy[1,:,:,:], gm,
                       pf_vel=pf_vel_t,
                       show_label=!stimuli)
        end

        if !isnothing(tracker_masks)
            _render_tracker_masks(tracker_masks[t])
        end
        
        array ? push!(imgs, image_as_matrix()) : finish()
    end
    
    # tracking while dots are moving
    for t=1:k
        print("render timestep: $t \r")
        _init_drawing(t+freeze_time, path, gm, prefix)

        if !stimuli
            _draw_text("$t", [gm.area_width/2 - 100, gm.area_height/2 - 100], size=50)
        end
        
        if !isnothing(dot_positions)
            _render_dots(dot_positions[t], gm, show_label=!stimuli)
            if !isnothing(probes)
                _render_probes(dot_positions[t], probes[t,:], gm)
            end
        end
    
        if !isnothing(pf_xy)
            attended_t = isnothing(attended) ? nothing : attended[t]
            pf_vel_t = isnothing(pf_vel) ? nothing : pf_vel[t,:,:,:]
            _render_pf(pf_xy[t,:,:,:], gm;
                       attended=attended_t,
                       pf_vel=pf_vel_t,
                       show_label=!stimuli)
        end

        if !isnothing(tracker_masks)
            _render_tracker_masks(tracker_masks[t])
        end

        array ? push!(imgs, image_as_matrix()) : finish()
    end
    # final freeze time showing the answer
    for t=1:freeze_time
        _init_drawing(t+k+freeze_time, path, gm, prefix)
        if !isnothing(dot_positions)
            _render_dots(dot_positions[k], gm;
                         highlighted=highlighted,
                         highlighted_color="blue",
                         show_label=!stimuli)
        end
        if !isnothing(pf_xy)
            attended_t = isnothing(attended) ? nothing : attended[k]
            pf_vel_t = isnothing(pf_vel) ? nothing : pf_vel[k,:,:,:]
            _render_pf(pf_xy[k,:,:,:], gm;
                       attended=attended_t,
                       pf_vel=pf_vel_t,
                       show_label=!stimuli)
        end
        if !isnothing(tracker_masks)
            _render_tracker_masks(tracker_masks[t])
        end
        array ? push!(imgs, image_as_matrix()) : finish()
    end
    array && return imgs
end

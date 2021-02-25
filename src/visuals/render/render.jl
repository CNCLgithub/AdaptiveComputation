using Luxor, ImageMagick

include("helpers.jl")

probe_color = "#a0a0a0" # in case we need it

function render_object(object::Object)
    error("not defined")
end

function render_object(dot::Dot; dot_color="#e0b388")
    _draw_circle(dot.pos[1:2], dot.radius, dot_color)
end

function render_polygon(cg::CausalGraph, p::Int64)
    vs = vertices(cg, p)
    length(vs) == 1 && return

    positions = @>> vs begin
        map(v -> get_prop(cg, v, :object))
        map(o -> get_pos(o))
    end

    inds = @>> 1:length(positions)-1 begin
        map(i -> (i, i+1))
    end
    push!(inds, (1, length(positions)))

    @>> inds begin
        foreach(ind -> _draw_arrow(positions[ind[1]][1:2], positions[ind[2]][1:2],
                                   "black", opacity=1.0, linewidth=1.0,
                                   arrowheadlength=0.0))
    end
end

function render_polygons(cg::CausalGraph)
    ps = filter_vertices(cg, (g,v) -> get_prop(g, v, :object) isa Polygon)
    @>> ps begin
        foreach(p -> render_polygon(cg, p))
    end
end

function render_force_edge(cg::CausalGraph, edge)
    vs = [src(edge), dst(edge)]
    positions = @>> vs begin
        map(v -> get_prop(cg, v, :object))
        map(o -> get_pos(o))
    end
    force = get_prop(cg, edge, :force)
    force_mag = norm(force)/10

    positions[2] .+= fill(1e-3, 3) # so that positions[1] != positions[2]

    _draw_arrow(positions[1][1:2], positions[2][1:2],
                "black", opacity=1.0, linewidth=force_mag,
                arrowheadlength=0.0)
end

function render_forces(cg::CausalGraph)
    edges = filter_edges(cg, :force) 
    @>> edges foreach(e -> render_force_edge(cg, e))
end


"""
    renders the causal graph
"""
function render_cg(cg::CausalGraph, gm::AbstractGMParams;
                   show_label=true,
                   highlighted=Bool[],
                   show_polygons=false,
                   show_walls=false,
                   show_polygon_centroids=false,
                   show_forces=false,
                   show_labels)
    
    if show_walls
        walls = get_objects(cg, Wall)
        @>> walls foreach(w -> _draw_arrow(w.p1, w.p2, "black", arrowheadlength=0.0))
    end

    # this renders the polygon connections between dots
    show_polygons && render_polygons(cg)

    dots = get_objects(cg, Dot)

    # furthest (highest z) comes first in depth_perm
    depth_perm = sortperm(map(x -> x.pos[3], dots), rev=true)
    
    for i in depth_perm
        dot_color = !isempty(highlighted) && highlighted[i] ? "#ea3433" : "#b4b4b4"

        render_object(dots[i], dot_color=dot_color)
        if show_labels
            _draw_text("$i", dots[i].pos[1:2] .+ [dots[i].width/2, dots[i].height/2])
        end
    end
    
    # this just renders the centroid
    if show_polygon_centroids
        polygons = get_objects(cg, Polygon)
        @>> polygons foreach(p -> _draw_circle(get_pos(p)[1:2], 10.0, "blue"))
    end
    
    # this renders the forces between walls, polygons and dots
    show_forces && render_forces(cg)
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
function _render_tracker_masks(tracker_masks, gm;
                               tracker_masks_colors=["indigo", "green", "blue", "yellow"])

    n_particles, n_trackers = size(tracker_masks)

    for p=1:n_particles
        for i=1:n_trackers
            opacity = 0.5 * _get_opacity(p,n_particles)
            _draw_array(tracker_masks[p,i], gm, tracker_masks_colors[i], opacity=opacity)
        end
    end
end

"""
    renders particle filter inferred positions (and velocities) of the tracked objects
    from the causal graph
"""
function render_pf(causal_graph, gm;
                   attended=nothing,
                   pf_color="darkslateblue",
                   show_label=true,
                   render_pos=true,
                   render_vel=false)
    
    n_particles = length(causal_graph)

    for p=1:n_particles
        objects = causal_graph[p].elements

        for i=1:length(objects)
            # drawing the predicted position of this tracker in particle
            if render_pos
                pred_pos = objects[i].pos
                _draw_circle(pred_pos, gm.dot_radius/3, pf_color, opacity=1.0)
            end
            
            if render_vel
                pred_vel = objects[i].vel
                _draw_arrow(pred_pos, pred_pos[1:2] + pred_vel*5, pf_color,
                            opacity=1.0)
            end
            
            if show_label
                _draw_text("$i", pred_pos)
            end
            
            # visualizing attention
            if !isnothing(attended)
                att_opacity = attended[i] * _get_opacity(p, n_particles)
                _draw_circle(pred_pos, 2.5*gm.dot_radius, "red", opacity=att_opacity, style=:stroke)
            end
        end
    end
end

function render_frame(t, path, gm;
                      gt_causal_graph=nothing,
                      pf_causal_graph=nothing,
                      pf_masks=nothing,
                      receptive_fields=nothing,
                      receptive_fields_overlap=nothing,
                      highlighted=Bool[],
                      show_polygons=false,
                      show_time=false,
                      show_labels=false,
                      show_forces=false)

    _init_drawing(t, path, gm,
                  receptive_fields = receptive_fields,
                  receptive_fields_overlap = receptive_fields_overlap)

    if show_time
        _draw_text("$t", [gm.area_width/2 - 100, gm.area_height/2 - 100], size=50)
    end
    
    if !isnothing(gt_causal_graph)
        render_cg(gt_causal_graph, gm;
                  show_labels=show_labels,
                  show_forces=show_forces,
                  highlighted=highlighted,
                  show_polygons=show_polygons)
    end

    if !isnothing(pf_causal_graph)
        render_pf(pf_causal_graph, gm)
                  #attended = isnothing(attended) ? nothing : attended[t])
    end

    if !isnothing(pf_masks)
        _render_tracker_masks(pf_masks, gm)
    end
    
    finish()
end


"""
    renders detailed information about inference on top of stimuli

    gm - generative model parameters

    optional:
    freeze_time - time before and after movement (for highlighting targets and querying)
    highlighted - array 
    receptive_fields - Tuple{Int} specifying how many tiles in the x and y dimensions
"""
function render(gm, k;
                gt_causal_graphs=nothing,
                pf_causal_graphs=nothing,
                pf_masks=nothing,
                freeze_time=0,
                highlighted_start=Bool[],
                highlighted_end=highlighted_start,
                # attended=nothing, TODO add attention to the pf_causal_graphs
                receptive_fields=nothing,
                receptive_fields_overlap=0.0,
                show_time=false,
                show_forces=false,
                show_labels=false,
                show_polygons=false,
                show_polygon_centroids=false,
                path="render")
    
    ispath(path) || mkpath(path)

    # initial freeze indicating the targets
    @>> 1:freeze_time begin
        foreach(t -> render_frame(t, path, gm,
                                  gt_causal_graph=gt_causal_graphs[1],
                                  receptive_fields=receptive_fields,
                                  receptive_fields_overlap=receptive_fields_overlap,
                                  highlighted=highlighted_start,
                                  show_polygons=true))
    end
   
    # movement
    @>> 1:k begin
        foreach(t -> render_frame(t+freeze_time, path, gm,
                                  gt_causal_graph=gt_causal_graphs[t],
                                  receptive_fields=receptive_fields,
                                  receptive_fields_overlap=receptive_fields_overlap,
                                  show_polygons=show_polygons,
                                  show_forces=show_forces,
                                  show_time=show_time,
                                  show_labels=show_labels))
    end

    # initial freeze indicating the targets
    @>> 1:freeze_time begin
        foreach(t -> render_frame(t+k+freeze_time, path, gm,
                                  gt_causal_graph=gt_causal_graphs[k],
                                  receptive_fields=receptive_fields,
                                  receptive_fields_overlap=receptive_fields_overlap,
                                  highlighted=highlighted_end,
                                  show_polygons=show_polygons))
    end
end

export render

export visualize_inference

function visualize_inference(results, gt_causal_graphs, gm, attention, path;
                             render_tracker_masks=false,
                             render_model=false,
                             render_map=false,
                             masks=nothing)
    extracted = extract_chain(results)
    causal_graphs = extracted["unweighted"][:causal_graph]
    
    if render_map
        for t=1:size(causal_graphs,1)
            for i=1:size(causal_graphs, 2)
                trace = extracted["unweighted"][:trace][t,i]
                causal_graphs[t,i] = extract_causal_graph(trace)[1]
            end
        end
    end

    k = size(causal_graphs, 1)
    tracker_masks = render_tracker_masks ? extracted["unweighted"][:tracker_masks] : nothing
    
    # aux_state = extracted["aux_state"]
    # attention_weights = [aux_state[t].stats for t = 1:k]
    # attention_weights = collect(hcat(attention_weights...)')
    
    # plot_compute_weights(attention_weights, path)

    # attempts = Vector{Int}(undef, k)
    # attended = Vector{Vector{Float64}}(undef, k)
    # for t=1:k
        # attempts[t] = aux_state[t].attempts
        # attended[t] = aux_state[t].attended_trackers
    # end
    # MOT.plot_attention(attended, attention.sweeps, path)
    # plot_rejuvenation(attempts, path)
    
    println(path)
    # visualizing inference on stimuli
    render(gm, k;
           gt_causal_graphs=gt_causal_graphs,
           causal_graphs=causal_graphs,
           #attended=attended/attention.sweeps,
           tracker_masks=tracker_masks,
           path = joinpath(path, "render"))
    
    render_model || return

    # also rendering from the perspective of the model
    tracker_masks = extracted["unweighted"][:tracker_masks]
    masks = []
    for i=1:k
        push!(masks,[])
        for j=1:gm.n_trackers
            push!(masks[i], mask(tracker_masks[i,1,j]))
        end
    end
    full_imgs = get_full_imgs(masks)
    model_render(full_imgs, gm,
                 # model_cgs=causal_graphs,
                 path=joinpath(path, "model_render"))
end


function model_render(full_imgs, gm;
                      model_cgs=nothing,
                      path="model_render")

    h, w, ah, aw = gm.img_height, gm.img_width, gm.area_height, gm.area_width
    
    k = length(full_imgs)
    
    for t=1:k
        img = full_imgs[t]
    
        if !isnothing(model_cgs)
            for cg in model_cgs[t,:]
                positions = map(x->x.pos[1:2], cg.elements)
                for pos in positions
                    x, y = translate_area_to_img(pos[1], pos[2], h, w, ah, aw)
                    draw_circle!(img, [x,y], 3.0, false)
                    draw_circle!(img, [x,y], 2.0, true)
                    draw_circle!(img, [x,y], 1.0, false)
                end
            end
        end
    
        mkpath(path)
        filename = "$(lpad(t, 3, "0")).png"
        save(joinpath(path, filename), img)
    end
end

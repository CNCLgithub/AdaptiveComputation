export visualize_inference

function make_series(gm, gt_cgs, pf_cgs, rf_dims, attended::Vector{Vector{Float64}},
                     padding::Int64;
                     base = "/renders/painter_test")

    @unpack area_width, area_height = gm
    gt_targets = [fill(true, gm.n_trackers); fill(false, Int64(gm.distractor_rate))]
    pf_targets = fill(true, gm.n_trackers)
    
    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    nt = length(pf_cgs)

    for i = 1:nt
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[i])


        p = RFPainter(area_dims = (area_height, area_width),
                      rf_dims = rf_dims)
        MOT.paint(p, gt_cgs[i])


        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[i])

        p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                          KinPainter())
        MOT.paint(p, pf_cgs[i])


        p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                          IDPainter(colors = ["purple", "green", "blue", "yellow"],
                                    label = true))
        MOT.paint(p, pf_cgs[i])

        p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
                                     dims = (gm.area_height, gm.area_width))
        MOT.paint(p, pf_cgs[i], attended[i])

        finish()
    end
end


function visualize_inference(results, gt_causal_graphs, gm, rf_dims, attention, path;
                             render_tracker_masks=false,
                             render_model=false,
                             render_map=false,
                             masks=nothing,
                             padding = 3)

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
    #tracker_masks = render_tracker_masks ? extracted["unweighted"][:tracker_masks] : nothing
    
    aux_state = extracted["aux_state"]
    attention_weights = [aux_state[t].stats for t = 1:k]
    attention_weights = collect(hcat(attention_weights...)')

    plot_compute_weights(attention_weights, path)

    attempts = Vector{Int}(undef, k)
    attended = Vector{Vector{Float64}}(undef, k)
    for t=1:k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end
    MOT.plot_attention(attended, attention.sweeps, path)
    plot_rejuvenation(attempts, path)
    
    # visualizing inference on stimuli
    pf_cgs = @>> 1:size(causal_graphs, 1) map(i -> causal_graphs[i,1])
    padding
    make_series(gm, gt_causal_graphs, pf_cgs, rf_dims, attended,
                padding;
                base = joinpath(path, "render"))

    
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

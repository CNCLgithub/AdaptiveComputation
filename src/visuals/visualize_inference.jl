export visualize_inference

function make_series(gm, gt_cgs, pf_cgs, padding::Int64;
                     base = "/renders/painter_test")

    @unpack area_width, area_height = gm
    gt_targets = [fill(true, gm.n_trackers); fill(false, Int64(gm.distractor_rate))]
    pf_targets = fill(true, gm.n_trackers)
    
    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    nt = length(pf_cgs)
    
    series = Vector{Vector{CausalGraph}}(undef, padding + nt)
    painters = Vector{Vector{Vector{Painter}}}(undef, padding + nt)
    # gt_series = Vector{CausalGraph}(undef, padding + nt)
    # gt_painters = Vector{Vector{Painter}}(undef, padding + nt)

    # pf_series = Vector{CausalGraph}(undef, padding + nt)
    # pf_painters = Vector{Vector{Painter}}(undef, padding + nt)

    frame = 1
    for i = 1:padding
        # gt_series[i] = gt_cgs[1]
        # pf_series[i] = pf_cgs[1]
        series[i] = [gt_cgs[1], pf_cgs[1]]

        gt_painters = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width)),
            PsiturkPainter(),
            TargetPainter(targets = gt_targets),
        ]
        pf_painters = [
            SubsetPainter(cg -> only_targets(cg, pf_targets),
                          IDPainter())
        ]

        painters[i] = [gt_painters, pf_painters]
        frame += 1
    end
    for i = 1:nt
        # gt_series[i + padding] = gt_cgs[i]
        # pf_series[i + padding] = pf_cgs[i]
        
        series[i + padding] = [gt_cgs[i], pf_cgs[i]]

        gt_painters = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width)),
            PsiturkPainter(),
        ]
        pf_painters = [
            SubsetPainter(cg -> only_targets(cg, pf_targets),
                          KinPainter()),
            SubsetPainter(cg -> only_targets(cg, pf_targets),
                          IDPainter())
        ]
        frame += 1

        painters[i + padding] = [gt_painters, pf_painters]
    end
    
    #paint_series([gt_series, pf_series], [gt_painters, pf_painters])
    paint_series(series, painters)
end

function visualize_inference(results, gt_causal_graphs, gm, attention, path;
                             render_tracker_masks=false,
                             render_model=false,
                             render_map=false,
                             masks=nothing,
                             receptive_fields=nothing,
                             receptive_fields_overlap = 0)
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
    
    println(path)
    
    pf_cgs = @>> 1:size(causal_graphs, 1) map(i -> causal_graphs[i,1])
    make_series(gm, gt_causal_graphs, pf_cgs, 3;
                base = joinpath(path, "render"))

    # visualizing inference on stimuli
    # render(gm, k;
           # gt_causal_graphs=gt_causal_graphs,
           # causal_graphs=causal_graphs,
           # attended=attended/attention.sweeps,
           # #tracker_masks=tracker_masks,
           # path = joinpath(path, "render"),
           # receptive_fields=receptive_fields,
           # receptive_fields_overlap = receptive_fields_overlap)
    
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

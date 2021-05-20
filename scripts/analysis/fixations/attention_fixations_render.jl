using MOT
using JLD2
using FileIO
using Luxor
using Lazy: @>>

function paint_fixations(gm, gt_cgs, pf_cgs,
                         fixations, attended;
                         padding = 1,
                         base = "/renders/fixations")

    isdir(base) && rm(base, recursive=true)
    mkpath(base)

    MOT.@unpack area_width, area_height = gm
    nt = length(gt_cgs)
    
    frame = 1
    pf_targets = fill(true, gm.n_trackers)

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[1])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[1])
        
        p = TargetPainter(targets = gm.targets)
        MOT.paint(p, gt_cgs[1])

        finish()
        
        frame += 1
    end

    for i = 1:nt
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[i])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[i])

        # p = TargetPainter(targets = gm.targets)
        # MOT.paint(p, gt_cgs[i])
    
        p = FixationsPainter()
        a = max(1, i-10)
        MOT.paint(p, fixations[a:i, :, :])

        # p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
                                     # dims = (50, 37))
        # MOT.paint(p, pf_cgs[i][end], attended[i])

        # attention center
        p = AttentionCentroidPainter(tau = 1.0,
                                     opacity = 0.3,
                                     color="red")
        MOT.paint(p, pf_cgs[i][end], attended[i])

        # geometric centroid
        p = AttentionCentroidPainter(tau = 0.0,
                                     opacity = 0.3,
                                     color="blue")
        MOT.paint(p, pf_cgs[i][end], attended[i])

        for (j, pf_cg) in enumerate(pf_cgs[i])
            p = SubsetPainter(cg -> MOT.only_targets(cg, pf_targets),
                              IDPainter(colors = ["purple", "green", "blue", "yellow"],
                                        label = false,
                                        alpha = j/length(pf_cgs[i])))
            MOT.paint(p, pf_cg)

            p = SubsetPainter(cg -> MOT.only_targets(cg, pf_targets),
                              KinPainter(alpha = j/length(pf_cgs[i])))
            MOT.paint(p, pf_cg)
        end
        
        p = AttentionRingsPainter()
        MOT.paint(p, pf_cgs[i][end], attended[i])



        finish()
        frame += 1
    end

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[nt])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[nt])

        p = TargetPainter(targets = gm.targets)
        MOT.paint(p, gt_cgs[nt])
        
        finish()
        frame += 1
    end
end


function render_fixations(scene_number, results,
                          fps, time;
                          experiment_path = "/experiments/fixations_target_designation",
                          fixations_subjects_path = "output/fixations/trial_fixations.jld2",
                          fixations_dataset_path = "output/datasets/fixations_dataset.jld2",
                          fpsdataset = 60)

    frames_per_step = round(Int64, fpsdataset / fps)
    last_frame = round(Int64, time * fpsdataset)
    
    fixations = load(fixations_subjects_path)["trial_fixations"][scene_number, 1:frames_per_step:last_frame, :, :]

    scene_data = MOT.load_scene(scene_number, fixations_dataset_path)
    cgs = scene_data[:gt_causal_graphs][1:frames_per_step:last_frame]
    aux_data = scene_data[:aux_data]

    extracted = extract_chain(results)
    causal_graphs = extracted["unweighted"][:causal_graph]
    k = size(causal_graphs, 1)
    aux_state = extracted["aux_state"]
    # attention_weights = [aux_state[t].stats for t = 1:k]
    # attention_weights = collect(hcat(attention_weights...)')
    attempts = Vector{Int}(undef, k)
    attended = Vector{Vector{Float64}}(undef, k)
    for t=1:k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end

    gm = HGMParams(area_height = 586,
                  area_width = 800,
                  dot_radius = 15,
                  targets = aux_data[:targets])

    traces = extracted["unweighted"][:trace]
    pf_cgs = @>> traces[:,1] map(trace -> MOT.get_n_back_cgs(trace, 1))
    paint_fixations(gm, cgs, pf_cgs, fixations, attended)
end


# render_fixations(1)

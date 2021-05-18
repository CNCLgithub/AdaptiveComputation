using MOT
using JLD2
using FileIO
using Luxor

function paint_fixations(gm, gt_cgs, pf_cgs,
                         fixations, attended;
                         padding = 1,
                         base = "/renders/fixations")

    isdir(base) && rm(base, recursive=true)
    mkpath(base)

    MOT.@unpack area_width, area_height = gm
    nt = length(gt_cgs)
    
    frame = 1

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

        p = TargetPainter(targets = gm.targets)
        MOT.paint(p, gt_cgs[i])
    
        p = FixationsPainter()
        MOT.paint(p, fixations[i, :, :])

        # p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
        #                                 dims = (gm.area_height, gm.area_width),
        #                                 attention_color = "blue")
        # MOT.paint(p, pf_cgs[i][end], fill(0.25, 4))

        # attention center
        p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
                                     dims = (50, 37))
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


function render_fixations(scene_number, chain;
                          experiment_path = "/experiments/fixations_target_designation",
                          fixations_subjects_path = "output/fixations/trial_fixations.jld2",
                          fixations_dataset_path = "output/datasets/fixations_dataset.jld2")

    fixations = load(fixations_subjects_path)["trial_fixations"][scene_number, :, :, :]
    scene_data = MOT.load_scene(scene_number, fixations_dataset_path)

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

    cgs = scene_data[:gt_causal_graphs]
    gm = HGMParams(area_height = 586,
                  area_width = 800,
                  dot_radius = 15,
                  targets = aux_data[:targets])
    # gm = scene_data[:gm]

    traces = extracted["unweighted"][:trace]
    pf_cgs = @>> traces[:,1] map(trace -> MOT.get_n_back_cgs(trace, 1))
    paint_fixations(gm, cgs, pf_cgs, fixations, attended)
end


# render_fixations(1)

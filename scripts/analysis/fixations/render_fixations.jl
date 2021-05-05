using MOT
using JLD2
using FileIO
using Luxor

function paint_fixations(gm, gt_cgs, fixations;
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
        MOT.paint(p, fixations[i,:,:])
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


function render_fixations(scene_number;
                          fixations_subjects_path = "output/data/fixations/parsed_fixations/trial_fixations.jld2",
                          fixations_dataset_path = "output/datasets/fixations_dataset.jld2")

    fixations = load(fixations_subjects_path)["trial_fixations"][scene_number, :, :, :]
    scene_data = MOT.load_scene(scene_number, fixations_dataset_path, GMParams();
                                generate_masks=false)

    cgs = scene_data[:gt_causal_graphs]
    gm = scene_data[:gm]

    display(gm)
    
    paint_fixations(gm, cgs, fixations)
end


render_fixations(1)

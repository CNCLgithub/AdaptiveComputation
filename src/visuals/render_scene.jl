export render_scene

function render_scene(gm, gt_cgs, pf_cgs, rf_dims, attended::Matrix{Float64};
                     base = "/renders/render_scene")
    @unpack area_width, area_height = gm

    gt_targets = [fill(true, gm.n_trackers); fill(false, Int64(gm.distractor_rate))]
    pf_targets = fill(true, gm.n_trackers)
    
    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    np, nt = size(pf_cgs)

    att_rings = AttentionRingsPainter(max_attention = maximum(attended))

    for i = 1:nt
        print("rendering scene... timestep $i / $nt \r")
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width),
                        background = "white")
        MOT.paint(p, gt_cgs[i])
        
        p = RFPainter(area_dims = (area_height, area_width),
                      rf_dims = rf_dims)
        MOT.paint(p, gt_cgs[i])

        p = PsiturkPainter(dot_color = "black")
        MOT.paint(p, gt_cgs[i])

        nj = length(pf_cgs[i])
        alpha = 3.0 * 1.0 / nj
        for j = 1:np
            p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                              KinPainter(alpha = alpha))
            MOT.paint(p, pf_cgs[j, i])

            p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                              IDPainter(colors = TRACKER_COLORSCHEME[:],
                                        label = false,
                                        alpha = alpha))
            MOT.paint(p, pf_cgs[j, i])
        end

        p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                          KinPainter())
        MOT.paint(p, pf_cgs[end, i])

        MOT.paint(att_rings, pf_cgs[end, i], attended[:, i])
        

        """
        # geometric center
        p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
                                     dims = (gm.area_height, gm.area_width),
                                     attention_color = "blue")
        MOT.paint(p, pf_cgs[i][end], fill(0.25, 4))

        # attention center
        p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
                                     dims = (gm.area_height, gm.area_width))
        MOT.paint(p, pf_cgs[i][end], attended[i])
        """

        finish()
    end
end

"""
    to render just ground truth elements
"""
function render_scene(gm, gt_cgs, targets;
                      padding = 3,
                      base = "/renders/render_scene")

    isdir(base) && rm(base, recursive=true)
    mkpath(base)

    @unpack area_width, area_height = gm
    nt = length(gt_cgs)
    
    frame = 1

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[1])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[1])
        
        p = TargetPainter(targets = targets)
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

        finish()
        frame += 1
    end

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[nt])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[nt])

        p = TargetPainter(targets = targets)
        MOT.paint(p, gt_cgs[nt])
        
        finish()
        frame += 1
    end
end

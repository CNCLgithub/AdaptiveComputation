export render_scene


function render_scene(gm, gt_cgs, pf_cgs, rf_dims, attended::Vector{Vector{Float64}};
                      base = "/renders/render_scene")

    @unpack area_width, area_height = gm

    gt_targets = [fill(true, gm.n_trackers); fill(false, Int64(gm.distractor_rate))]
    pf_targets = fill(true, gm.n_trackers)
    
    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    nt = length(pf_cgs)

    for i = 1:nt
        print("rendering scene... timestep $i / $nt \r")
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, gt_cgs[i])

        p = RFPainter(area_dims = (area_height, area_width),
                      rf_dims = rf_dims)
        MOT.paint(p, gt_cgs[i])


        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[i])
        
        for (j, pf_cg) in enumerate(pf_cgs[i])
            # p = SubsetPainter(cg -> only_targets(cg, pf_targets),
            #                   KinPainter(alpha = j/length(pf_cgs[i])))
            # MOT.paint(p, pf_cg)

            p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                              IDPainter(colors = ["purple", "green", "blue", "yellow"],
                                        label = false,
                                        alpha = j/length(pf_cgs[i])))
            MOT.paint(p, pf_cg)
        end


        p = SubsetPainter(cg -> only_targets(cg, pf_targets),
                          KinPainter())
        MOT.paint(p, pf_cgs[i][end])


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

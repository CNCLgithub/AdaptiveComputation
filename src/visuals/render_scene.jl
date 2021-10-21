export render_scene

red = Colors.color_names["red"]

function render_scene(gm::GMParams,
                      gt_cgs::Vector{CausalGraph},
                      pf_st::Matrix{InertiaKernelState},
                      attended::Matrix{Float64};
                      base = "/renders/render_scene",
                      max_attention = 600.)
    @unpack area_width, area_height = gm

    nx = size(attended, 1)

    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    np, nt = size(pf_st)

    alpha = 3.0 * 1.0 / np
    att_rings = AttentionRingsPainter(max_attention = max_attention)

    for i = 1:nt
        print("rendering scene... timestep $i / $nt \r")

        # first render gt state of observed objects
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width),
                        background = "white")
        MOT.paint(p, gt_cgs[i])
        
        p = PsiturkPainter(dot_color = "black")
        MOT.paint(p, gt_cgs[i])

        p = SubsetPainter(cg -> only_targets(gt_cgs[i]),
                          IDPainter(colors = fill(red, nx)))

        # then render each particle's state
        for j = 1:np
            p = SubsetPainter(cg -> only_targets(cg),
                              KinPainter(alpha = alpha))
            @unpack world = (pf_st[j, i])
            MOT.paint(p, world)


            c = correspondence(pf_st[j, i])
            tweights = vec(sum(attended[:, i] .* c, dims = 1))
            MOT.paint(att_rings, world, tweights)

            nt = length(tweights)
            # @show nt
            nt === 0 && continue
            p = SubsetPainter(cg -> only_targets(cg),
                              IDPainter(colors = TRACKER_COLORSCHEME[fill(nt, nx)],
                              # IDPainter(colors = fill(red, nx),
                              # IDPainter(colors = TRACKER_COLORSCHEME[:],
                                        label = false,
                                        alpha = 0.5))
            MOT.paint(p, world)


        end

        # p = SubsetPainter(cg -> only_targets(cg, pf_targets),
        #                   KinPainter())
        # MOT.paint(p, pf_cgs[end, i])




        # # geometric center
        # p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
        #                              dims = (gm.area_height, gm.area_width),
        #                              attention_color = "blue")
        # MOT.paint(p, pf_cgs[i][end], fill(0.25, 4))

        # # attention center
        # p = AttentionGaussianPainter(area_dims = (gm.area_height, gm.area_width),
        #                              dims = (gm.area_height, gm.area_width))
        # MOT.paint(p, pf_cgs[i][end], attended[i])


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

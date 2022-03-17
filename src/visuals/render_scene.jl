export render_scene

red = Colors.color_names["red"]

function render_scene(gm::GMParams,
                      gt_cgs::Vector{CausalGraph},
                      pf_st::Matrix{InertiaKernelState},
                      attended::Dict;
                      base = "/renders/render_scene",
                      max_attention = 600.)
    @unpack area_width, area_height = gm

    isdir(base) && rm(base, recursive=true)
    mkpath(base)
    np, nt = size(pf_st)

    maxatt = @>> attended values collect(Vector{Int64}) mapreduce(maximum, max)
    alpha = 3.0 * 1.0 / np
    for i = 1:nt
        print("rendering scene... timestep $i / $nt \r")

        # first render gt state of observed objects
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width),
                        background = "white")
        MOT.paint(p, gt_cgs[i])


        step = 1
        steps = max(1, i-7):i
        for k = steps
            alpha = exp(0.5 * (k - i))
            p = SubsetPainter(cg -> only_targets(cg, BitVector([0,0,0,0,1,1,1,1])),
                              PsiturkPainter(dot_color = "black",
                                             alpha = alpha))
            MOT.paint(p, gt_cgs[k])
        end

        # p = SubsetPainter(cg -> only_targets(gt_cgs[i]),
        #                   IDPainter(colors = [],
        #                             label = true))
        # p = IDPainter(colors = [], label = true)
        MOT.paint(p, gt_cgs[i])
        # then render each particle's state
        for j = 1:np
            p = SubsetPainter(cg -> only_targets(cg),
                              KinPainter(alpha = alpha))
            @unpack world = (pf_st[j, i])
            MOT.paint(p, world)


            # tw = target_weights(pf_st[j, i], attended[:, i])
            att = attended[i => j]
            # att_rings = AttentionRingsPainter(max_attention = maxatt, # sum(att),
            #                                   opacity = 0.8,
            #                                   radius = 40.,
            #                                   linewidth = 7.0,
            #                                   attention_color = "red")
            # MOT.paint(att_rings, world, att)

            nt = length(att)
            # @show nt
            nt === 0 && continue
            step = 1
            steps = max(1, i-7):i
            for k = steps
                @unpack world = (pf_st[j, k])
                alpha = 0.5 * exp(0.5 * (k - i))
                p = SubsetPainter(cg -> only_targets(cg),
                                  IDPainter(colors = map(x -> parse(RGB, x),
                                                         ["#A3A500","#00BF7D","#00B0F6","#E76BF3"]),
                                            label = false,
                                            alpha = alpha))
                MOT.paint(p, world)
                step += 1
            end
        end
        finish()
    end
end

"""
    to render just ground truth elements
"""
function render_scene(dims::Tuple{Float64, Float64}, gt_cgs::Vector{CausalGraph}, targets;
                      padding = 3,
                      base = "/renders/render_scene")

    isdir(base) && rm(base, recursive=true)
    mkpath(base)

    nt = length(gt_cgs)
    
    frame = 1

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = dims)
        MOT.paint(p, gt_cgs[1])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[1])

        p = IDPainter(colors = [], label = true)
        # p = TargetPainter(targets = targets)
        MOT.paint(p, gt_cgs[1])

        finish()
        
        frame += 1
    end

    for i = 1:nt
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = dims)
        MOT.paint(p, gt_cgs[i])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[i])
        p = IDPainter(colors = [], label = true)
        MOT.paint(p, gt_cgs[i])

        finish()
        frame += 1
    end

    for i = 1:padding
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = dims)
        MOT.paint(p, gt_cgs[nt])

        p = PsiturkPainter()
        MOT.paint(p, gt_cgs[nt])

        p = IDPainter(colors = [], label = true)
        # p = TargetPainter(targets = targets)
        MOT.paint(p, gt_cgs[nt])
        
        finish()
        frame += 1
    end
end

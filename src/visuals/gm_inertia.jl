export render_trace, render_pf

color_codes = parse.(RGB, ["#A3A500","#00BF7D","#00B0F6","#E76BF3"])

function render_gstate!(canvas, d::Dot, c)
    @unpack gstate = d
    rows = rowvals(gstate)
    vs = nonzeros(gstate)
    m, n = size(gstate)
    for j = 1:n
        for i in nzrange(gstate, j)
            y = rows[i]
            # canvas[y, j] = RGBA{Float64}(c.r, c.g, c.b, vs[i])
            canvas[y, j] = ColorBlendModes.blend(canvas[y,j],
                                 RGBA{Float64}(c.r, c.g, c.b, vs[i]))
        end
    end
    return nothing
end

function render_prediction!(canvas, gm::InertiaGM, st::InertiaState)
    @unpack objects = st
    ne = length(objects)
    for i = 1:ne
        color_code = RGB{Float64}(color_codes[i])
        render_gstate!(canvas, objects[i], color_code)
    end
    return nothing
end

function render_observed!(canvas, gm::InertiaGM, st::InertiaState;
                          alpha::Float64 = 0.6)
    @unpack xs = st
    nx = length(xs)
    color_code = RGBA{Float64}(1., 1., 1., alpha)
    for i = 1:nx
        canvas[xs[i]] .= ColorBlendModes.blend.(canvas[xs[i]], color_code)
    end
    return nothing
end

function render_trace(gm::InertiaGM,
                      tr::Gen.Trace,
                      path::String)


    @unpack img_dims = gm

    (init_state, states) = get_retval(tr)
    t = first(get_args(tr))

    isdir(path) && rm(path, recursive=true)
    mkdir(path)

    for tk = 1:t
        st = states[tk]
        canvas = fill(RGBA{Float64}(0., 0., 0., 1.0), img_dims)
        render_prediction!(canvas, gm, st)
        render_observed!(canvas, gm, st)
        save("$(path)/$(tk).png", canvas)
    end

    return nothing
end

function render_pf(gm::InertiaGM,
                   chain::SeqPFChain,
                   path::String)

    @unpack state, auxillary = chain
    @unpack img_dims = gm

    isdir(path) && rm(path, recursive=true)
    mkdir(path)

    np = length(state.traces)
    tr = first(state.traces)
    t = first(get_args(tr))
    states = collect(map(x -> last(get_retval(x)), state.traces))
    for tk = 1:t
        canvas = fill(RGBA{Float64}(0., 0., 0., 1.0), img_dims)
        for p = 1:np
            render_prediction!(canvas, gm, states[p][tk])
        end
        render_observed!(canvas, gm, states[1][tk])
        save("$(path)/$(tk).png", canvas)
    end
    return nothing
end

red = Colors.color_names["red"]

function paint(p::InitPainter, st::InertiaState)
    height, width = p.dimensions
    Drawing(width, height, p.path)
    Luxor.origin()
    background(p.background)
end
function MOT.paint(p::Painter, st::InertiaState)
    for o in st.objects
        paint(p, o)
    end
    return nothing
end
function MOT.paint(p::IDPainter, st::InertiaState)
    for i in eachindex(st.objects)
        paint(p, st.objects[i], i)
    end
    return nothing
end
function MOT.paint(p::AttentionRingsPainter,
                   st::InertiaState,
                   weights::Vector{Float64})
    ne = length(st.objects)
    for i = 1:ne
        paint(p, st.objects[i], weights[i])
    end
    return nothing
end
function render_scene(gm::InertiaGM,
                      gt_states::Vector{InertiaState},
                      pf_st::Matrix{InertiaState},
                      attended::Matrix{Float64};
                      base::String)
    @unpack area_width, area_height = gm

    isdir(base) && rm(base, recursive=true)
    mkdir(base)
    np, nt = size(pf_st)

    alpha = 3.0 * 1.0 / np
    for i = 1:nt
        print("rendering scene... timestep $i / $nt \r")

        # first render gt state of observed objects
        p = InitPainter(path = "$base/$i.png",
                        dimensions = (area_height, area_width),
                        background = "white")
        MOT.paint(p, gt_states[i])


        # paint gt
        step = 1
        steps = max(1, i-7):i
        for k = steps
            alpha = exp(0.5 * (k - i))
            p = PsiturkPainter(dot_color = "black",
                               alpha = alpha)
            MOT.paint(p, gt_states[k])
        end
        p = IDPainter(colors = [], label = true)
        MOT.paint(p, gt_states[i])

        # then render each particle's state
        for j = 1:np

            # paint motion vectors
            p = KinPainter(alpha = alpha)
            pf_state = pf_st[j, i]
            MOT.paint(p, pf_state)

            # attention rings
            # tw = target_weights(pf_st[j, i], attended[:, i])
            att_rings = AttentionRingsPainter(max_attention = 1.0,
                                              opacity = 0.8,
                                              radius = 40.,
                                              linewidth = 7.0,
                                              attention_color = "red")
            MOT.paint(att_rings, pf_state, attended[:, i])

            # add tails
            step = 1
            steps = max(1, i-7):i
            for k = steps
                alpha = 0.5 * exp(0.5 * (k - i))
                p = IDPainter(colors = color_codes,
                              label = false,
                              alpha = alpha)
                MOT.paint(p, pf_st[j, k])
                step += 1
            end
        end
        finish()
    end
end

# """
#     to render just ground truth elements
# """
# function render_scene(dims::Tuple{Float64, Float64}, gt_states::Vector{CausalGraph}, targets;
#                       padding = 3,
#                       base = "/renders/render_scene")

#     isdir(base) && rm(base, recursive=true)
#     mkpath(base)

#     nt = length(gt_states)

#     frame = 1

#     for i = 1:padding
#         p = InitPainter(path = "$base/$frame.png",
#                         dimensions = dims)
#         MOT.paint(p, gt_states[1])

#         p = PsiturkPainter()
#         MOT.paint(p, gt_states[1])

#         p = IDPainter(colors = [], label = true)
#         # p = TargetPainter(targets = targets)
#         MOT.paint(p, gt_states[1])

#         finish()

#         frame += 1
#     end

#     for i = 1:nt
#         p = InitPainter(path = "$base/$frame.png",
#                         dimensions = dims)
#         MOT.paint(p, gt_states[i])

#         p = PsiturkPainter()
#         MOT.paint(p, gt_states[i])
#         p = IDPainter(colors = [], label = true)
#         MOT.paint(p, gt_states[i])

#         finish()
#         frame += 1
#     end

#     for i = 1:padding
#         p = InitPainter(path = "$base/$frame.png",
#                         dimensions = dims)
#         MOT.paint(p, gt_states[nt])

#         p = PsiturkPainter()
#         MOT.paint(p, gt_states[nt])

#         p = IDPainter(colors = [], label = true)
#         # p = TargetPainter(targets = targets)
#         MOT.paint(p, gt_states[nt])

#         finish()
#         frame += 1
#     end
# end

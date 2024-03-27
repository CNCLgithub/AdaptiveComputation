export render_trace, render_pf, render_scene

using FileIO: save

color_codes = parse.(RGB, ["#A3A500","#00BF7D","#00B0F6","#E76BF3"])

function render_gstate!(canvas, d::Dot, c, aw)
    @unpack gstate = d
    iw,ih = size(canvas)
    m = zeros(ih, iw)
    nt = length(d.gstate)
    for t = 1:nt
        gc = gstate[t]
        @inbounds for i = 1:iw, j = 1:ih
            x = SVector{2, Float64}([(i - 0.5*iw) *  aw / iw,
                                    (j - 0.5*ih) * -aw / ih])
            v = exp(Gen.logpdf(mvnormal, x, gc.mu, gc.cov) + gc.w + 4.)
            v = min(1.0, v)
            m[j, i] = max(m[j,i], v)
        end
    end
    # m .*= 1.0/nt
    @inbounds for i = eachindex(canvas)
        c = RGBA{Float64}(c.r, c.g, c.b, m[i])
        canvas[i] = ColorBlendModes.blend(canvas[i], c)
    end
    return nothing
end

function render_prediction!(canvas, gm::InertiaGM, st::InertiaState)
    @unpack objects = st
    ne = length(objects)
    for i = 1:ne
        color_code = RGB{Float64}(color_codes[i])
        render_gstate!(canvas, objects[i], color_code, gm.area_width)
    end
    return nothing
end

function render_observed!(canvas, gm::InertiaGM, st::InertiaState;
                          alpha::Float64 = 1.0)
    @unpack xs = st
    nx = length(xs)
    color_code = RGBA{Float64}(1., 1., 1., alpha)
    @inbounds for i = 1:nx
        xt = xs[i]
        nt = length(xt)
        for j = 1:nt
            a,b = xt[j]
            x,y = translate_area_to_img(a,b,gm.img_width, gm.area_height)
            canvas[y,x] = color_code
        end
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


function render_trial(gm::InertiaGM,
                      states::Vector{InertiaState},
                      path::String)
    @unpack img_dims = gm

    t = length(states)
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
                   chain::PFChain,
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

function MOT.paint(p::Painter, st::GMState)
    for o in st.objects
        paint(p, o)
    end
    return nothing
end
function MOT.paint(p::Union{IDPainter,KinPainter}, st::GMState)
    for i in eachindex(st.objects)
        paint(p, st.objects[i], i)
    end
    return nothing
end

function MOT.paint(gm::GenerativeModel, st::GMState)
    return nothing
end

function MOT.paint(p::AttentionRingsPainter,
                   st::GMState,
                   weights::Vector{Float64})
    ne = min(length(weights), length(st.objects))
    for i = 1:ne
        paint(p, st.objects[i], weights[i])
    end
    return nothing
end
function render_scene(gm::T,
        gt_states::Vector{<:U},
        pf_st::Matrix{<:U},
        attended::Vector;
        base::String) where {T<:GenerativeModel, U<:GMState{T}}
    @unpack area_width, area_height = gm

    isdir(base) && rm(base, recursive=true)
    mkdir(base)
    np, nt = size(pf_st)

    alpha = 3.0 * 1.0 / np

    att_rings = AttentionRingsPainter(max_attention = 1.0,
                                    opacity = 1.00,
                                    radius = 40.,
                                    linewidth = 15.0,
                                    attention_color = "red")
    # att_centroid = AttentionCentroidPainter()

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
            p = KinPainter(color = color_codes,
                           alpha = 0.1,
                           tail = true)
            pf_state = pf_st[j, i]
            for i = 1:length(pf_state.objects)
                obj = pf_state.objects[i]
                if target(obj) > 0
                    MOT.paint(p, obj, i)
                end
            end
            MOT.paint(gm, pf_state)
            # MOT.paint(p, pf_state)

            # attention rings
            MOT.paint(att_rings, pf_state, attended[i])
            # MOT.paint(att_centroid, pf_state, attended[:, i])

        end
        finish()
    end
    return nothing
end

function render_scene(gm::GenerativeModel,
                      gt_states::AbstractVector{T},
                      base::String) where {T<:GMState}
    @unpack area_width, area_height = gm

    isdir(base) && rm(base, recursive=true)
    mkdir(base)

    nt = length(gt_states)
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
        colors = fill("black", length(gt_states[i].objects))
        for obj_i = 1:length(colors)
            if MOT.target(gt_states[i].objects[obj_i]) > 0.
                colors[obj_i] = "red"
            end
        end
        p = IDPainter(colors = colors, label = true)
        MOT.paint(p, gt_states[i])
        finish()
    end
    return nothing
end

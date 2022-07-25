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

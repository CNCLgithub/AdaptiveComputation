using Gen
using GenRFS
using MOT
using Setfield
using Parameters

function render_dot(d::Dot, gr::Graphics)
    @unpack img_width, img_height = gr
    area_width, area_height = (800., 800.)

    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos[1:2]...,
                                 img_height, img_width,
                                 area_width, area_height)
    scaled_r = d.radius/area_width*img_width # assuming square
    space = MOT.exp_dot_mask(x, y, scaled_r, gr)

    @unpack nlog_bernoulli, flow_decay_rate = gr
    flow = MOT.ExponentialFlow(decay_rate = flow_decay_rate, memory = space)
    space = flow.memory
    e = LogBernoulliElement{BitMatrix}(nlog_bernoulli, mask, (space,))
    es = RFSElements{BitMatrix}(undef, 1)
    es[1] = e
    es
end

function main()

    gr = Graphics(;
                  img_width = 100,
                  img_height = 100,
                  inner_f = 0.75,
                  inner_p = 0.99,
                  outer_f = 5.0,
                  outer_p = 0.9,
                  flow_decay_rate = -1.0,
                  nlog_bernoulli = -200)
    d0 = MOT.Dot() # dot at center
    d0_mask = render_dot(d0, gr)

    # gri = gr
    gri = @set gr.outer_f = 1.1

    for i = 1:10
        # move dot along x axis
        x = (i-1) * 6
        di = MOT.Dot(;pos = [0, x, 0.])
        di_mask = render_dot(di, gri)
        di_obs = rfs(di_mask)

        lpdf = Gen.logpdf(rfs, di_obs, d0_mask)
        println("Logpdf of $(x): $(lpdf)")

    end
end

main();

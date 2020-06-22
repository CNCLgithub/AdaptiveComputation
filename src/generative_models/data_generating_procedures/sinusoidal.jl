export dgp

function dgp(k::Int, params::GMMaskParams,
             period::Float64)
    # assumes two trackers, one distractor

    positions = zeros(k+1, 3, 3)
    # distractor is fixed at -x
    # one target is fixed at +x
    amp = params.area_width * 0.25
    positions[:, 3, 1] .= -amp
    positions[:, 2, 1] .= amp
    for t = 1:k+1
        positions[t, 1, 1] = sin(period * (t-1) * pi / 180) * amp * 0.95
    end
    init_pos = positions[1,:,:]
    masks = get_masks(positions[2:end, : ,:], params.dot_radius, params.img_height,
                      params.img_width, params.area_height, params.area_width)
    return init_pos, masks
end

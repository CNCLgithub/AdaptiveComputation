using Gen
using MOT
using Profile
using StatProfilerHTML

function main()
    gm = GMParams()
    dm = InertiaDynamics()
    rf_dims = (1,1)
    img_dims = (100, 100)
    receptive_fields = get_rectangle_receptive_fields(rf_dims,
                                                      img_dims,
                                                      1E-10, # threshold
                                                      0.0,   # overlap
                                                      )
    graphics = Graphics(;
                        flow_decay_rate = -0.3,
                        rf_dims = rf_dims,
                        img_dims = img_dims,
                        receptive_fields = receptive_fields)
    args = (3, gm, dm, graphics)

    Profile.init(delay = 0.001,
                 n = 10^6)
    @profilehtml trace, _ = generate(gm_inertia_mask, args)
    @profilehtml trace, _ = generate(gm_inertia_mask, args)
end

main();

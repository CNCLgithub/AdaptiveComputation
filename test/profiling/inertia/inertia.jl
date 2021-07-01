using Gen
using MOT
using Profile
using StatProfilerHTML

function main()
    gm = GMParams()
    dm = InertiaModel()
    rf_dims = (4,4)
    # rf_dims = (1,1)
    img_dims = (200, 200)
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
    args = (10, gm, dm, graphics)

    Profile.init(delay = 1E-4,
                 n = 10^8)
    # @profilehtml trace, _ = generate(gm_inertia_mask, args)
    @profilehtml trace, _ = generate(gm_inertia_mask, args)
    @time trace, _ = generate(gm_inertia_mask, args)
end

main();

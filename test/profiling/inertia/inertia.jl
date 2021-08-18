using Gen
using MOT
using Profile
using StatProfilerHTML
using Lazy: @>, @>>
using Setfield
using UnicodePlots: heatmap

function main()
    gm = GMParams(;n_trackers = 1,
                  distractor_rate = 1.)
    dm = InertiaModel(;w_min = 0.2,
                      w_max = 0.2)
    rf_dims = (4,4)
    # rf_dims = (1,1)
    img_dims = (200, 200)
    receptive_fields = get_rectangle_receptive_fields(rf_dims,
                                                      img_dims,
                                                      1E-10, # threshold
                                                      0.0,   # overlap
                                                      )
    graphics = Graphics(;
                        bern_existence_prob = 0.9,
                        flow_decay_rate = -0.3,
                        gauss_amp = 0.95,
                        gauss_std = 0.7,
                        gauss_r_multiple = 4.5,
                        rf_dims = rf_dims,
                        img_dims = img_dims,
                        receptive_fields = receptive_fields)
    args = (7, gm, dm, graphics)

    generate(gm_inertia_mask, args);
    @time trace, _ = generate(gm_inertia_mask, args)
    # @profilehtml trace, _ = generate(gm_inertia_mask, args)

    _, cgs = get_retval(trace)

    att = MapSensitivity(
        ancestral_steps = 3,
        objective = MOT.target_designation_receptive_fields,
    )
    constraints = MOT.get_init_constraints(cgs[1])
    masks = MOT.get_bit_masks_rf(collect(MOT.CausalGraph, cgs),
                                 graphics,
                                 gm)
    observations = MOT.get_observations(graphics, masks)

    graphics1 = @set graphics.gauss_std = 0.7
    @show graphics1

    tr, w = generate(gm_inertia_mask, (0, gm, dm, graphics1),
                     constraints)

    Profile.init(delay = 1E-6,
                 n = 10^7)
    Profile.clear()
    @profilehtml for (t, o) in enumerate(observations)
        tr, w, _, _ = update(tr, (t, gm, dm, graphics1),
                             (UnknownChange(), NoChange(),
                              NoChange(), NoChange()),
                             o)
        # @show w
        steps = 20
        accepted = 0
        for i = 1:steps
            new_tr, ls = MOT.tracker_kernel(tr, 1, att)
            # @show ls
            if log(rand()) < ls
                tr = new_tr
                accepted += 1
            end
        end
        println("acceptance ratio $(accepted / steps)")
    end


    # constraints = constraints_from_cgs(cgs, gm_inertia_mask, args)
    # choices = get_choices(trace)
    # masks = choices[:kernel => 1 => :receptive_fields => 1 => :masks]
    # display(sparse(masks[end]))

    # _, states = get_retval(trace)
    # display(get_prop(states[1], 6, :object))

    # masks = constraints[:kernel => 3 => :receptive_fields => 1 => :masks]
    # # display(sparse(masks[end]))
    # # display(constraints)
end

main();

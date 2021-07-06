using Gen
using MOT
using MetaGraphs
using SparseArrays
using LinearAlgebra
using Profile
using StatProfilerHTML

function constraints_from_cgs(cgs, gm, args)
    t = length(cgs)
    cm = MOT.get_init_constraints(cgs[1])
    prev_objects = MOT.get_objects(cgs[1], Dot)
    for i = 1:t
        objects = MOT.get_objects(cgs[i], Dot)
        for j = 1:length(objects)
            pos = objects[j].pos[1:2]
            delta = pos - prev_objects[j].pos[1:2]
            nd = norm(delta)
            ang = delta ./ nd
            ang = nd == 0. ? 0. : atan(ang[2], ang[1])
            cm[:kernel => i => :dynamics => :trackers => j => :mag] = nd
            cm[:kernel => i => :dynamics => :trackers => j => :ang] = ang
        end
        prev_objects = objects
    end

    # display(cm)
    trace, _ = generate(gm, args, cm)
    choices = get_choices(trace)

    observations = choicemap()
    for i = 1:t
        addr = :kernel => i => :receptive_fields
        set_submap!(observations, addr, get_submap(choices, addr))
    end
    return observations
end

function main()
    gm = GMParams(;n_trackers = 1,
                  distractor_rate = 1.)
    dm = InertiaModel()
    # rf_dims = (4,4)
    rf_dims = (1,1)
    img_dims = (200, 200)
    receptive_fields = get_rectangle_receptive_fields(rf_dims,
                                                      img_dims,
                                                      1E-10, # threshold
                                                      0.0,   # overlap
                                                      )
    graphics = Graphics(;
                        bern_existence_prob = 1.0,
                        flow_decay_rate = -0.3,
                        rf_dims = rf_dims,
                        img_dims = img_dims,
                        receptive_fields = receptive_fields)
    args = (45, gm, dm, graphics)

    Profile.init(delay = 1E-4,
                 n = 10^8)
    generate(gm_inertia_mask, args);
    @time trace, _ = generate(gm_inertia_mask, args)
    @profilehtml trace, _ = generate(gm_inertia_mask, args)

    _, cgs = get_retval(trace)
    constraints = constraints_from_cgs(cgs, gm_inertia_mask, args)
    choices = get_choices(trace)
    masks = choices[:kernel => 1 => :receptive_fields => 1 => :masks]
    display(sparse(masks[end]))

    _, states = get_retval(trace)
    display(get_prop(states[1], 6, :object))

    # masks = constraints[:kernel => 3 => :receptive_fields => 1 => :masks]
    # # display(sparse(masks[end]))
    # # display(constraints)
end

main();

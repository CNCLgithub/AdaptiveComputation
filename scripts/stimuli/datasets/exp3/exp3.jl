using Gen
using MOT
using JSON
using Accessors

default_gm = ISRGM(;
                   dot_repulsion = 30.0,
                   wall_repulsion = 30.0,
                   distance_factor = 100.0,
                   rep_inertia = 0.25,
                   max_distance = 150.0,
                   dot_radius = 20.0,
                   area_width = 800.0,
                   area_height = 800.0)

function trial_constraints(gm::ISRGM)
    cm = choicemap()
    for i = 1:gm.n_dots
        cm[:init_state => :init_kernel => i => :target] =
            (i <= gm.n_targets)
    end
    return cm
end

function trial_data(tr::Gen.Trace)
    steps, gm = get_args(tr)
    init_state, states = get_retval(tr)
    positions = []
    for t in 1:steps
        objects = states[t].objects
        time_step = []
        for i in 1:gm.n_dots
            push!(time_step, get_pos(objects[i]))
        end
        push!(positions, time_step)
    end
    trial = Dict(
        :positions => positions,
        :aux_data => (targets = Bool.([i <= gm.n_targets for i = 1:gm.n_dots]),
                      vel = gm.vel,
                      n_distractors = gm.n_dots - gm.n_targets)
    )
end

function main()
    steps = 240 # number of steps, 10s
    total_objects = 12
    n_targets = 1:6
    velocities = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    dataset = Dict[]
    for nt = n_targets, v = velocities
        gm = setproperties(default_gm,
                           (n_dots = total_objects,
                            n_targets = nt,
                            vel = v))
        cm = trial_constraints(gm)
        trace, _ = generate(gm_isr, (steps, gm), cm)
        data = trial_data(trace)
        push!(dataset, data)
    end

    open("/spaths/datasets/exp3.json", "w") do f
        write(f, json(dataset))
    end


end


main();

using Gen
using MOT
using JSON
using Accessors

default_gm = ISRGM(;
                   dot_repulsion = 50.0,
                   wall_repulsion = 25.0,
                   distance_factor = 100.0,
                   rep_inertia = 0.15,
                   max_distance = 125.0,
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
        push!(positions, map(get_pos, objects))
    end
    trial = Dict(
        :positions => positions,
        :aux_data => (targets = Bool.([i <= gm.n_targets for i = 1:gm.n_dots]),
                      vel = gm.vel,
                      n_distractors = gm.n_dots - gm.n_targets)
    )
end

function main()
    nscenes = 80
    steps = 120 # number of steps, 5s
    total_objects = 10
    ntargets = 3
    vel = 8.0
    dataset = Dict[]
    for _ = 1:nscenes
        gm = setproperties(default_gm,
                           (n_dots = total_objects,
                            n_targets = ntargets,
                            vel = vel))
        cm = trial_constraints(gm)
        trace, _ = generate(gm_isr, (steps, gm), cm)
        data = trial_data(trace)
        push!(dataset, data)
    end
    open("/spaths/datasets/exp3_localization_error.json", "w") do f
        write(f, json(dataset))
    end
end

main();

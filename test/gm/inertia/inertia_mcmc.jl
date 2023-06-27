using Gen
using MOT
using Profile
using BenchmarkTools
using StatProfilerHTML

using Random
Random.seed!(1234)

function main()

    gm = load(InertiaGM, "/project/test/gm/inertia/gm.json")
    steps = 10
    args = (steps, gm)
    cm = choicemap()
    for i = 1:4
        cm[:init_state => :trackers => i => :target] = true
    end

    # generate a trace to create observations
    tr, _ = generate(gm_inertia, args, cm)
    choices = get_choices(tr)

    # new trace to change via mcmc
    new_tr, _ = generate(gm_inertia, (0, gm),
                         get_submap(choices, :init_state))
    Profile.init(delay = 1E-6,
                 n = 10^7)
    Profile.clear()

    t = 5
    obs = choicemap()
    obs[:kernel => t => :masks] = choices[:kernel => t => :masks]
    new_tr, _ = update(new_tr, (t, gm), (UnknownChange(), NoChange()), obs)

    display(@benchmark MOT.tracker_kernel($new_tr, 1, 3))
    steps = 100
    accepted = 0
    @profilehtml for i = 1:steps
        _tr, w = MOT.tracker_kernel(new_tr, 1, 3)
        # @show ls
        if log(rand()) < w
            new_tr = _tr
            accepted += 1
        end
    end
    println("acceptance ratio $(accepted / steps)")
end

main();

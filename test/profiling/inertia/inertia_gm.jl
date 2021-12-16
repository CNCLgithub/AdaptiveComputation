using Gen
using MOT
using Profile
using StatProfilerHTML
using BenchmarkTools

function main()
    gm = GMParams(;death_rate = 0, max_things = 8)
    dm = InertiaModel()
    graphics = Graphics(img_width = 100,
                        img_height = 100,
                        flow_decay_rate = -0.75,
                        inner_f = 1.0,
                        inner_p = 0.95,
                        outer_f = 4.5,
                        outer_p = 0.6,
                        nlog_bernoulli = -200)
    rf_dims = (1,1)
    img_dims = (100, 100)
    args = (100, gm, dm, graphics)

    cm = choicemap()
    cm[:init_state => :n_trackers] = 4
    for i = 1:4
        cm[:init_state => :trackers => i => :target] = true
    end
    println("initial run for JIT")
    generate(gm_inertia_mask, args)
    @time generate(gm_inertia_mask, args)
    # Profile.clear_malloc_data()
    println("benchmark")
    Profile.init(delay = 0.0001,
                 n = 10^8)
    # generate(gm_inertia_mask, args);
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100
    # display(@benchmark generate($gm_inertia_mask, $args));
    # Profile.init(delay = 0.0001,
                 # n = 10^7)
    Profile.clear()
    # Profile.clear_malloc_data()
    println("profiling")
    @profilehtml for _ = 1:20
        generate(gm_inertia_mask, args)
    end
end

main();

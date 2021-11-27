using Gen
using MOT
using Profile
using StatProfilerHTML
using BenchmarkTools

function main()
    gm = GMParams(;death_rate = 0, max_things = 8)
    dm = InertiaModel()
    rf_dims = (1,1)
    img_dims = (100, 100)
    args = (100, gm, dm, graphics)

    cm = choicemap()
    cm[:init_state => :n_trackers] = 4
    for i = 1:4
        cm[:init_state => :trackers => i => :target] = true
    end
    println("initial run for JIT")
    @time generate(gm_inertia_mask, args)
    # Profile.clear_malloc_data()
    println("benchmark")
    @btime generate($gm_inertia_mask, $args);
    Profile.init(delay = 0.0001,
                 n = 10^7)
    Profile.clear()
    # Profile.clear_malloc_data()
    println("profiling")
    @profilehtml for _ = 1:1
        generate(gm_inertia_mask, args)
    end
end

main();

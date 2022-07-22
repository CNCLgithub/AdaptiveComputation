using Gen
using MOT
using Profile
using StatProfilerHTML
using BenchmarkTools

function main()
    gm = InertiaModel()
    # 100 steps
    args = (100, gm)
    cm = choicemap()
    for i = 1:4
        cm[:init_state => :trackers => i => :target] = true
    end

    println("initial run for JIT")
    generate(gm_inertia_mask, args)

    println("Benchmark for: `gm_inertia`")
    display(@benchmark generate($gm_inertia, $args));

    Profile.init(delay = 1E-6,
                 n = 10^7)
    Profile.clear()
    println("profiling")
    @profilehtml generate(gm_inertia_mask, args)
end

main();

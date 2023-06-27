using Gen
using MOT
using Profile
using StatProfilerHTML
using BenchmarkTools
using Random
Random.seed!(1234)

function main()
    gm = load(InertiaGM, "/project/test/gm/inertia/gm.json")
    steps = 100
    args = (steps, gm)
    cm = choicemap()
    for i = 1:4
        cm[:init_state => :trackers => i => :target] = true
    end

    println("initial run for JIT")
    tr, _ = generate(gm_inertia, args, cm)
    render_trace(tr, "/spaths/test/gm_inertia_trace")

    println("Benchmark for: `gm_inertia`")
    display(@benchmark generate($gm_inertia, $args));

    Profile.init(delay = 1E-7,
                 n = 10^7)
    Profile.clear()
    println("profiling")
    @profilehtml generate(gm_inertia, args)
end

main();

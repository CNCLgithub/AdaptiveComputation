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

    # choices = get_choices(tr)

    # new_tr, _ = generate(gm_inertia, (0, gm),
    #                      get_submap(choices, :init_state))
    # for t = 1:10
    #     obs = choicemap()
    #     obs[:kernel => t => :masks] = choices[:kernel => t => :masks]
    #     @time new_tr, w = update(new_tr, (t, gm), (UnknownChange(), NoChange()), obs)
    #     @show w
    # end


    # cs = get_choices(tr)
    # display(get_submap(cs, :kernel => 1 => :trackers))

    println("Benchmark for: `gm_inertia`")
    display(@benchmark generate($gm_inertia, $args));

    Profile.init(delay = 1E-7,
                 n = 10^7)
    Profile.clear()
    Profile.clear_malloc_data()
    println("profiling")
    # generate(gm_inertia, args)
    @profilehtml generate(gm_inertia, args)
    # @profilehtml generate(gm_inertia, args)
end

main();

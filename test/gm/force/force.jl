using Gen
using MOT

function test()
    gm =  MOT.load(ForceGM, "$(@__DIR__)/gm.json")
    trace, ls = Gen.generate(gm_force, (100, gm))
    @show MOT.td_assocs(trace)
    (init, states) = get_retval(trace)
    render_scene(gm, states, "/spaths/test/force_gm")
    return nothing
end

test();

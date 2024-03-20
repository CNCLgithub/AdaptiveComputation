using MOT

experiment_name = "exp2_probes"
dataset = "/spaths/datasets/$(experiment_name).json"
render_out = "/spaths/datasets/$(experiment_name)/rendered"
img_dims = (800., 800.)

function render(gm, i::Int64)
    scene_data = MOT.load_scene(gm,
                                dataset,
                                i)
    gt_states = scene_data[:gt_states]
    render_scene(gm, gt_states,
                 joinpath(render_out, "$i"))
    return nothing
end

function main()
    n = 40
    try
        isdir(render_out) || mkpath(render_out)
    catch e
        println("could not make dir $(render_out)")
    end

    gm = MOT.load(InertiaGM, "$(@__DIR__)/gm.json")
    render(gm, 22)
    # for i = 1:40
    #     render(gm, i)
    # end
end

main();

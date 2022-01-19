using MOT: load_scene, render_scene

const experiment_name = "exp2_probes"
const dataset = "/spaths/datasets/$(experiment_name).json"
const render_out = "/spaths/renders/$(experiment_name)"
const img_dims = (800., 800.)

function render(i::Int64)
    scene_data = load_scene(dataset,i)
    targets = scene_data[:aux_data]["targets"]
    gt_cgs = scene_data[:gt_causal_graphs]
    render_scene(img_dims, gt_cgs, targets;
                 base = joinpath(render_out, "$i"))
    return nothing
end

function main()
    n = 40
    try
        isdir(render_out) || mkpath(render_out)
    catch e
        println("could not make dir $(render_out)")
    end
    foreach(render, 1:n)
end

main();

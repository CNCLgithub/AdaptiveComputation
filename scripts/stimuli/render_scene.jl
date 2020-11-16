using Base.Filesystem

using HDF5
using ArgParse
using MOT

function render(d_path::String, scene::Int64)

    scene_data = MOT.load_scene(scene, d_path, default_gm;
                                generate_masks=false)
    path = "/renders/closeup/$(scene)"

    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    cgs = scene_data[:gt_causal_graphs]

    MOT.render(default_gm, 480;
           gt_causal_graphs = cgs,
           path = path,
           stimuli = false,
           freeze_time = 24)

end

# function parse_commandline()
#     s = ArgParseSettings()

#     @add_arg_table s begin
#         "dataset"
#         help = "dataset to load scene info"
#         arg_type = String
#         required = true

#         "scene"
#         help = "dataset to load scene info"
#         arg_type = Int
#         required = true
#     end

#     return parse_args(s)
# end

# function main()
#     args = parse_commandline()
#     render(args["dataset"], args["scene"])
# end

# main()

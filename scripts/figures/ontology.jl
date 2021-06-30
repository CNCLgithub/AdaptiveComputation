using Luxor
using MOT
using MOT: only_targets
using Parameters: @unpack

dataset_path = joinpath("output", "datasets", "exp3_polygons_v3.jld2")
scene_id = 10
# scene_id = 13 # poly
scene_data = MOT.load_scene(scene_id, dataset_path, HGMParams();
                            generate_masks=false)
r_fields_dim = (5, 5)
gm = scene_data[:gm]
k = length(scene_data[:gt_causal_graphs])
gt_cgs = scene_data[:gt_causal_graphs]
targets = scene_data[:aux_data][:targets]


function make_series(gm, cgs, backtrace::Int64)

    @unpack area_width, area_height, targets = gm
    base = "/renders/ontology"
    isdir(base) || mkdir(base)
    nt = length(cgs)
    frame = 1
    for i = backtrace:nt
        p = InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width))
        MOT.paint(p, cgs[i])
        for j = 1 : backtrace
            p = SubsetPainter(cg -> only_targets(cg, targets),
                              IDPainter(colors = ["purple", "green", "blue", "yellow"],
                                        label = false,
                                        alpha = j / backtrace))
            MOT.paint(p, cgs[i - backtrace + j])
        end


        p = SubsetPainter(cg -> only_targets(cg, targets),
                          KinPainter())
        MOT.paint(p, cgs[i])

        # distractors
        p = SubsetPainter(cg -> only_targets(cg, .!(targets)),
                          PoissDotPainter())
        MOT.paint(p, cgs[i])

        finish()
        frame += 1
    end
    return nothing
end

make_series(gm, gt_cgs, 10)

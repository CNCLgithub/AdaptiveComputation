using MOT
using MOT: CausalGraph, InitPainter, IFPainter, PolyPainter,
    TargetPainter, IDPainter, SubsetPainter, PsiturkPainter,
    KinPainter,
    paint_series, only_targets
using Parameters: @unpack

dataset_path = joinpath("output", "datasets", "exp1_difficulty.jld2")
scene_id = 1
scene_data = MOT.load_scene(scene_id, dataset_path, HGMParams();
                            generate_masks=false)
r_fields_dim = (5, 5)
gm = scene_data[:gm]
k = length(scene_data[:gt_causal_graphs])
gt_cgs = scene_data[:gt_causal_graphs]
targets = scene_data[:aux_data][:targets]


function make_series(gm, cgs, padding::Int64)

    @unpack area_width, area_height, targets = gm
    targets = [fill(true, 4); fill(false, 4)]
    println(targets)

    base = "/renders/classic_trial_poster"
    isdir(base) || mkdir(base)
    nt = length(cgs)
    series = Vector{CausalGraph}(undef, 2 * padding + nt)
    painters = Vector{Vector{Painter}}(undef, 2 * padding + nt)
    frame = 1
    for i = 1:padding
        series[i] = cgs[1]
        painters[i] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width),
                        background = "white"),
            PsiturkPainter(dot_color = "black"),
            TargetPainter(targets = targets,
                          dot_radius_multiplier = 0.7)
        ]
        frame += 1
    end
    for i = 1:nt
        series[i + padding] = cgs[i]
        painters[i + padding] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width),
                        background = "white"),
            PsiturkPainter(dot_color = "black"),
            SubsetPainter(cg -> only_targets(cg, targets),
                          KinPainter()),
            # IFPainter(),
            # TargetPainter(targets = targets),
            # PolyPainter(),
            # SubsetPainter(cg -> only_targets(cg, targets),
            #               IDPainter())
        ]
        frame += 1
    end
    for i = 1:padding
        series[i + padding + nt] = cgs[nt]
        painters[i + padding + nt] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width),
                        background = "white"),
            PsiturkPainter(dot_color = "black"),
            TargetPainter(targets = targets,
                          dot_radius_multiplier = 0.7)
        ]
        frame += 1
    end
    paint_series(series, painters)
end

make_series(gm, gt_cgs, 48)

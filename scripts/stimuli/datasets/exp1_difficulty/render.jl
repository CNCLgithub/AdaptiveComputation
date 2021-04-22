using MOT
using MOT: CausalGraph, InitPainter, IFPainter, PolyPainter,
    TargetPainter, IDPainter, SubsetPainter, PsiturkPainter,
    KinPainter,
    paint_series, only_targets
using Parameters: @unpack

function make_series(gm, cgs, padding::Int64;
                     base = "/renders/painter_test")

    @unpack area_width, area_height, targets = gm
    isdir(base) || mkdir(base)
    nt = length(cgs)
    series = Vector{CausalGraph}(undef, padding + nt)
    painters = Vector{Vector{Painter}}(undef, padding + nt)
    frame = 1
    for i = 1:padding
        series[i] = cgs[1]
        painters[i] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width)),
            PsiturkPainter(),
            #IFPainter(),
            #PolyPainter(show_centroid=true),
            TargetPainter(targets = targets),
            # SubsetPainter(cg -> only_targets(cg, targets),
                          # IDPainter())
        ]
        frame += 1
    end
    for i = 1:nt
        series[i + padding] = cgs[i]
        painters[i + padding] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width)),
            PsiturkPainter(),
            # SubsetPainter(cg -> only_targets(cg, targets),
            #               KinPainter()),
            #IFPainter(),
            # TargetPainter(targets = targets),
            #PolyPainter(show_centroid=true),
            # SubsetPainter(cg -> only_targets(cg, targets),
                          # IDPainter())
        ]
        frame += 1
    end
    
    paint_series(series, painters)
end

println("running rendering script")

dataset_path = joinpath("output", "datasets", "exp1_difficulty.jld2")

file = MOT.jldopen(dataset_path, "r")
n_scenes = file["n_scenes"]
close(file)

for scene_id=1:n_scenes
    scene_data = MOT.load_scene(scene_id, dataset_path, HGMParams();
                                generate_masks=false)
    #r_fields_dim = (5, 5)
    gm = scene_data[:gm]
    k = length(scene_data[:gt_causal_graphs])
    gt_cgs = scene_data[:gt_causal_graphs]
    #targets = scene_data[:aux_data][:targets]

    make_series(gm, gt_cgs, 20;
                base = joinpath("/renders", "exp1_difficulty", "$scene_id"))
    # make_series(gm, gt_cgs, 0)
end


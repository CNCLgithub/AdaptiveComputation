using Gen
using MOT
using MOT: CausalGraph,
    paint_series, only_targets
using Parameters: @unpack

r_fields_dim = (5, 5)
scene_structure = fill(1, 16)
cm = Gen.choicemap()
for (j, s) in enumerate(scene_structure)
    cm[:init_state => :polygons => j => :n_dots] = s
end
hgm =  HGMParams(n_trackers = 16,
                distractor_rate = 0.0,
                area_width = 1200,
                area_height = 1200,
                init_pos_spread = 500,
                targets = [fill(1, 8); fill(0, 8)])
scene_data = dgp(240, hgm, SquishyDynamicsModel();
                generate_masks=false,
                cm=cm)

k = length(scene_data[:gt_causal_graphs])
gt_cgs = scene_data[:gt_causal_graphs]
# targets = scene_data[:aux_data][:targets]


function make_series(gm, cgs, padding::Int64)

    @unpack area_width, area_height, targets = gm
    base = "/renders/rf"
    isdir(base) || mkdir(base)
    nt = length(cgs)
    series = Vector{CausalGraph}(undef, padding + nt)
    painters = Vector{Vector{Painter}}(undef, padding + nt)
    frame = 1
    for i = 1:nt
        series[i + padding] = cgs[i]
        painters[i + padding] = [
            InitPainter(path = "$base/$frame.png",
                        dimensions = (area_height, area_width)),
            PsiturkPainter(),
            RFPainter(area_dims = (area_height, area_width),
                      rf_dims = (5,5))
            # SubsetPainter(cg -> only_targets(cg, targets),
            #               KinPainter()),
            # IFPainter(),
            # TargetPainter(targets = targets),
            # PolyPainter(),
            # SubsetPainter(cg -> only_targets(cg, targets),
            #               IDPainter())
        ]
        frame += 1
    end
    paint_series(series, painters)
end

make_series(hgm, gt_cgs,0)

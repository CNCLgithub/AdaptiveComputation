using MOT
using Gen
using Random
using MetaGraphs
using LightGraphs
Random.seed!(5)

dataset_path = joinpath("/datasets", "exp3_polygons_v3_instructions.jld2")
ff_k = 10 # fast forward at the beginning to get non-overlapping
k = 240
min_distance = 45.0
area_width = 1200.0
area_height = 1200.0
init_pos_spread = 450.0

dm = SquishyDynamicsModel()

polygons = collect(3:3)
vertices = collect(3:4)

pvs = Iterators.product(polygons, vertices)
cms = ChoiceMap[]
aux_data = Any[]
gms = HGMParams[]
ff_ks = Int64[] # how much to fastforward in each scene to have non-overlapping positions

# for each polygon-vertice pair there are 4 scenes:
# 2 with polygon structure
# 1 with independent dots with matched init position
# 1 with independent dots with random init position
for pv in pvs
    p = pv[1] 
    v = pv[2] 
    println("\n\n\ngenerating $p polygons, $v vertices scenario\n")

    scene_structure = fill(v, 2*p)
    targets = Bool[fill(1, p*v); fill(0, p*v)]
   
    # polygons
    cm = Gen.choicemap()
    for (j, s) in enumerate(scene_structure)
        cm[:init_state => :polygons => j => :n_dots] = s
    end
    push!(gms, HGMParams(n_trackers = 2*p,
                         distractor_rate = 0.0,
                         area_width = area_width,
                         area_height = area_height,
                         init_pos_spread = init_pos_spread,
                         targets = targets))
    ad = (scene_structure = scene_structure,
          targets = targets)
    push!(aux_data, ad)
    
    # generating one scene to constrain to same positions
    # the polygon scene and independent dots scene
    tries = 0
    while true
        tries += 1
        # running only a little bit to get the constraint for the initial config
        global scene_data = dgp(ff_k+1, gms[end], dm;
                                generate_masks=false,
                                cm=cm,
                                generate_cm=true)
        forward_scene_data!(scene_data, ff_k)
        md = is_min_distance_satisfied(scene_data, min_distance)
        di = are_dots_inside(scene_data, gms[end])
        println("tries $tries, md=$md, di=$di")
        md && di && break
    end
    
    # constraining the polygon scene
    gt_cg = first(scene_data[:gt_causal_graphs])
    push!(cms, init_constraint_from_cg(gt_cg, scene_data[:cm]))

    # constraining the individual dots scene
    init_pos = @>> MOT.get_objects(gt_cg, MOT.Polygon) map(x -> MOT.get_pos(x))
    cm = choicemap()
    n_trackers = p*2
    for i=1:n_trackers
        cm[:init_state => :polygons => i => :n_dots] = 1
        cm[:init_state => :polygons => i => :x] = init_pos[i][1]
        cm[:init_state => :polygons => i => :y] = init_pos[i][2]
        cm[:init_state => :polygons => i => :z] = init_pos[i][3]
    end
    push!(cms, cm)
    targets = Bool[fill(1, p); fill(0, p)]
    push!(gms, HGMParams(n_trackers = n_trackers,
                         distractor_rate = 0.0,
                         area_width = area_width,
                         area_height = area_height,
                         init_pos_spread = init_pos_spread,
                         targets = targets))
    ad = (scene_structure = fill(1, n_trackers),
          targets = targets)
    push!(aux_data, ad)

    # no fastforwarding on the scenes that already have constrained init state
    append!(ff_ks, [1, 1])
end

println("lengths: gms $(length(gms)), cms $(length(cms)), aux_data $(length(aux_data)), ff_ks $(length(ff_ks))")
MOT.generate_dataset(dataset_path, length(gms), k, gms, dm;
                     min_distance=min_distance,
                     cms=cms,
                     aux_data=aux_data,
                     ff_ks=ff_ks)

using JSON
using MOT
using MOT: CausalGraph, SimpleDiGraph, add_vertex!, MetaGraphs.nv, set_prop!
using MetaGraphs
using Lazy: @>, @>>


function frame_data_to_pos(frame_data;
                           area_width = 1500,
                           area_height = 1500,
                           target_w = 800,
                           target_h = 800)

    targets = get_targets(frame_data)
    n_objects = length(targets)
    pos = Vector{Vector{Float64}}(undef, n_objects)

    for i=1:n_objects
        x = frame_data["Object $(lpad(i, 2, "0")).x"]
        y = frame_data["Object $(lpad(i, 2, "0")).y"]

        x *= target_w/area_width
        y *= target_h/area_height

        x -= target_w/2
        y -= target_h/2

        pos[i] = [x,y]
    end
    
    return pos
end


function pos_to_dots(pos)
    @>> pos map(p -> Dot([p; 0.0], zeros(2)))
end

function dots_to_cg(dots)
    cg = CausalGraph(SimpleDiGraph())

    for dot in dots
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        set_prop!(cg, v, :object, dot)
    end

    return cg
end


function get_targets(frame_data)
    n_targets = frame_data["Number of Targets"]
    # max number of objects is 16
    # and the objects not present have 0 values
    n_distractors = @>> 1:16 begin
        map(i -> frame_data["Object $(lpad(i, 2, "0")).x"])
        filter(!iszero)
        length
        x -> x - n_targets
    end

    return [fill(true, n_targets); fill(false, n_distractors)]
end

function get_cg!(cgs::Vector{CausalGraph}, frame_data)
    frame_number = frame_data["Frame number"]
    
    cg = @> frame_data begin
        frame_data_to_pos
        pos_to_dots
        dots_to_cg
    end

    cgs[frame_number] = cg
end


# getting data from one subject
dir = joinpath("output", "data", "fixations", "MOT_json_files")
fn_start = "behavior test trajectories_0_0_random_with fixation from sub_"
fn = @>> readdir(dir) filter(fn -> occursin(fn_start, fn)) first
#sub_data = JSON.parsefile(joinpath(dir, fn))

dataset_path = joinpath("/datasets", "fixations_dataset.jld2")
n_scenes = 120 # scene = trial in this dataset
jldopen(dataset_path, "w") do file
    file["n_scenes"] = n_scenes
    
    for i=1:n_scenes
        print("working on scene $i \r")
        
        # filtering frames for this scene
        scene_data = @>> sub_data filter(d -> d["Trial number"] == i)

        scene = JLD2.Group(file, "$i")
        targets = get_targets(scene_data[1])
        scene["targets"] = targets
        scene["vel_deg_sec"] = scene_data[1]["Speed (deg/sec)"]
        
        cgs = Vector{CausalGraph}(undef, n_frames)
        foreach(frame_data -> get_cg!(cgs, frame_data), scene_data)
        # trying to access each 
        @>> 1:n_frames foreach(i -> cgs[i])

        scene["gt_causal_graphs"] = cgs
        scene["gm"] = HGMParams(area_width = 800,
                               area_height = 800,
                               targets = targets)
        scene["aux_data"] = (targets = targets,)
    end
end



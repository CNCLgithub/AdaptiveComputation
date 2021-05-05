using JSON
using JLD2
using MOT
using MOT: CausalGraph, SimpleDiGraph, add_vertex!, MetaGraphs.nv, set_prop!
using MetaGraphs
using Lazy: @>, @>>

include("helpers_constants.jl")

function frame_data_to_pos(frame_data)

    targets = get_targets(frame_data)
    n_objects = length(targets)
    pos = Vector{Vector{Float64}}(undef, n_objects)

    for i=1:n_objects
        x = frame_data["Object $(lpad(i, 2, "0")).x"]
        y = frame_data["Object $(lpad(i, 2, "0")).y"]

        pos[i] = translate(x, y)
    end
    
    return pos
end


function pos_to_dots(pos)
    @>> pos map(p -> Dot([p; 0.0], zeros(2), dot_radius))
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

"""
    returns Bool[] indicating which objects are targets
"""
function get_targets(frame_data)
    n_targets = frame_data["Number of Targets"]
    # max number of objects is 16
    # and the objects not present have 0 x, y position
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

function get_avg_vel(cgs::Vector{CausalGraph})::Float64
    n_frames = length(cgs)
    n_objects = length(MOT.get_objects(first(cgs), Dot))

    pos = Array{Float64}(undef, n_frames, n_objects, 2)
    for i=1:n_frames
        dots = MOT.get_objects(cgs[i], Dot)
        for j=1:n_objects
            pos[i,j,:] = dots[j].pos[1:2]
        end
    end

    pos_t0 = pos[1:end-1,:,:]
    pos_t1 = pos[2:end,:,:]
    delta_pos = pos_t1 - pos_t0
    
    @>> Iterators.product(1:n_frames-1, 1:n_objects) begin
        map(ij -> delta_pos[ij[1], ij[2], :])
        map(norm)
        mean
    end
end


# getting data from one subject
dir = joinpath("output", "data", "fixations", "MOT_json_files")
fn_start = "behavior test trajectories_0_0_random_with fixation from sub_"
fn = @>> readdir(dir) filter(fn -> occursin(fn_start, fn)) first
sub_data = JSON.parsefile(joinpath(dir, fn))

dataset_path = joinpath("/datasets", "fixations_dataset.jld2")
n_scenes = 120 # scene = trial in this dataset
n_frames = 600

jldopen(dataset_path, "w") do file
    file["n_scenes"] = n_scenes
    
    for i=1:n_scenes
        print("working on scene $i \r")
        
        # filtering frames for this scene
        scene_data = @>> sub_data filter(d -> d["Trial number"] == i)

        scene = JLD2.Group(file, "$i")
        targets = get_targets(scene_data[1])
        scene["targets"] = targets
        
        cgs = Vector{CausalGraph}(undef, n_frames)
        foreach(frame_data -> get_cg!(cgs, frame_data), scene_data)
        # trying to access each 
        @>> 1:n_frames foreach(i -> cgs[i])
        
        scene["gt_causal_graphs"] = cgs
        scene["gm"] = HGMParams(area_width = TARGET_W,
                               area_height = TARGET_H,
                               dot_radius = DOT_RADIUS,
                               targets = targets)
        scene["aux_data"] = (targets = targets,
                             vel_deg_sec = scene_data[1]["Speed (deg/sec)"],
                             vel_avg = get_avg_vel(cgs))
    end
end



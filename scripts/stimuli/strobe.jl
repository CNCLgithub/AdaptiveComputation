using JLD2
using FileIO
using VideoIO
using MOT

function place_strobes(positions, x::Int, k::Int)
    ps = fill(fill(-2000., size(positions[1])), size(positions,1))
    t = size(positions, 1)
    for i = 1:t
        ((i - 1) % (x + k)) < x ? ps[i] = positions[i] : continue
    end
    ps
end

function compile_video(path::String)
    imgnames = filter(x->occursin(".png",x), readdir(path))
    intstrings =  map(x->split(x,".")[1],imgnames)
    p = sortperm(parse.(Int,intstrings))
    imgstack = []
    for imgname in imgnames[p]
        push!(imgstack,load(joinpath(path, imgname)))
    end
    encodevideo("$(path).mp4", imgstack)
    imgstack = nothing
    return nothing
end

function render_strobe_trial(trial, path::String;
                             x::Int64 = 1, k::Int64 = 1)

    positions = place_strobes(trial["positions"], x, k)
    gm = trial["gm"]
    render(gm, dot_positions=positions, path=path,
            stimuli=true, freeze_time=50, highlighted=collect(1:4))

    compile_video(path)
    return nothing
end


function render_strobe_trials(dataset::String)
    out = "/renders/strobe"
    ispath(out) || mkpath(out)
    jldopen(dataset, "r") do file
        n_trials = min(file["n_trials"], 1)

        for k = 1:3, x = 1:3
            for i=1:n_trials
                positions = file["$i"]["positions"]
                gm = file["$i"]["gm"]
                path = joinpath(out, "$(x)_$(k)_$(i)")
                render_strobe_trial(file["$i"], path; x = x, k = k)
            end
        end
    end

end

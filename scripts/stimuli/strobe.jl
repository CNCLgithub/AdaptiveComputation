using FileIO
using VideoIO
using MOT

function place_strobes(positions, k::Int)
    ps = fill(fill(-2000., size(positions[1])), size(positions,1))
    t = size(positions, 1)
    for i = range(1, t, step = k)
        ps[i] = positions[i]
    end
    ps
end

function render_strobe_trial(trial, path::String; k::Int64 = 4)

    positions = place_strobes(trial["positions"], k)
    gm = trial["gm"]
    render(gm, dot_positions=positions, path=path,
            stimuli=true, freeze_time=50, highlighted=collect(1:4))

    compile_video(path)
    return nothing
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

function render_probe_trials(dataset::String)
    out = "/renders/strobe"
    ispath(out) || mkpath(out)
    jldopen(dataset, "r") do file
        n_trials = min(file["n_trials"], 5)

        for k = 1:6
            for i=1:n_trials
                positions = file["$i"]["positions"]
                gm = file["$i"]["gm"]
                path = joinpath(out, "$k", "$i")
                render_strobe_trial(file["$i"], path; k = k)
            end
        end
    end

end

using VideoIO

function compile_videos(img_out_path::String, videos_out_path::String)
    mkpath(videos_out_path)

    trials = readdir(img_out_path)
    for trial in trials
        quantiles = readdir(joinpath(img_out_path, trial))
        for quantile in quantiles
            path = joinpath(img_out_path, trial, quantile)
            imgnames = filter(x->occursin(".png",x), readdir(path))
            intstrings =  map(x->split(x,".")[1],imgnames)
            p = sortperm(parse.(Int,intstrings))
            imgstack = []
            for imgname in imgnames[p]
                push!(imgstack, load(joinpath(path, imgname)))
            end
            video_dir = joinpath(videos_out_path, trial)
            mkpath(video_dir)
            encodevideo(joinpath(video_dir, "$quantile.mp4"), imgstack)
        end
    end
end

using MOT

dataset_path = "output/datasets/isr_dataset.jld2"
render_path = "output/renders/isr_dataset/test/"
jldopen(dataset_path, "r") do file
    n_trials = file["n_trials"]
    for i=1:n_trials
        positions = file["$i"]["positions"]
        gm = file["$i"]["gm"]
        path = joinpath(render_path, "$i")
        render(gm, dot_positions=positions, path=path,
               stimuli=true, freeze_time=50, highlighted=collect(1:4))
    end
end

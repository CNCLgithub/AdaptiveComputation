using MOT
using JLD2


function render_dataset(dataset_path, render_path)
    jldopen(dataset_path, "r") do file
        n_trials = file["n_trials"]
        for i=1:n_trials
            positions = file["$i"]["positions"]
            gm = file["$i"]["gm"]
            path = joinpath(render_path, "$i")
            render(gm, dot_positions=positions, path=path,
                   stimuli=true, freeze_time=10, highlighted=collect(1:4))
        end
    end
end

dataset_path = "output/datasets/exp1.jld2"
render_path = "output/renders/exp1/"

render_dataset(dataset_path, render_path)

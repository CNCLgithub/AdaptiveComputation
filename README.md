# mot
Multiple object tracking repository in Julia.

This branch includes the basic data generating procedures (found under `src/generative_models/data_generating_procedures/`) and the basic renderer (found under `src/visuals/render.jl`).

## Setup and running
1. Clone repository `git clone https://github.com/CNCLgithub/mot` and `cd mot`
2. Switch to this branch `git checkout basic-dgp-render`
3. Run `./setup.sh cont_pull conda julia` to pull the container and setup Conda and Julia environments.
4. Enter `./run.sh julia` to get into Julia REPL
5. Enter `include("scripts/demo/demo.jl")` in the Julia REPL to see how you can generate a dataset and render it using the codebase. You can find the generated dataset under `output/datasets/isr_dataset.jld2` and the rendered frames under `output/renders/isr_dataset`.

The code includes two motion models (`ISRDynamics` and `BrownianDynamics`).

To generate movies of the scenes you can make the movie script executable `chmod +x scripts/movie.sh` and run `./scripts/movie.sh <path to folder with rendered frames>` (e.g. for the demo renders you would run `./scripts/movie.sh output/renders/isr_dataset/`). You can then find the movies in the same folder.

One useful script can be found under `scripts/convert_dataset_to_json.jl` that can convert the HDF5 dataset generated using the codebase to a JSON file with an array of object positions for each scene. For instance, we use the JSON datasets in our JavaScript interactive animated experiments.

Please do not hesitate to reach out if you need help with anything.

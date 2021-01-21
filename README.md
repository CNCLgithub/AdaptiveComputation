# mot
Multiple object tracking repository in Julia.

This branch includes the basic data generating procedures (found under `src/generative_models/data_generating_procedures/`) and the basic renderer (found under `src/visuals/render.jl`).

## Setup and running
1. Clone repository `git clone https://github.com/CNCLgithub/mot` and `cd mot`
2. Switch to this branch `git checkout basic-dgp-render`
3. Run `./setup.sh cont_pull conda julia` to pull the container and setup Conda and Julia environments.
4. Enter `./run.sh julia` to get into Julia REPL
5. Enter `include("scripts/demo/demo.jl")` in the Julia REPL to see how you can generate a dataset and render it using the codebase.

The code includes two motion models (`ISRDynamics` and `BrownianDynamics`).

One useful script can be found under `scripts/convert_dataset_to_json.jl` that can convert the HDF5 dataset generated using the codebase to a JSON file with an array of object positions for each scene. For instance, we use the JSON datasets in our JavaScript interactive animated experiments.

Please do not hesitate to reach out if you need help with anything.

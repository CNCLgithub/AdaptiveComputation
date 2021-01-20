# mot
Multiple object tracking repository in Julia.

This branch includes the basic data generating procedures (found under `src/generative_models/data_generating_procedures/`) and the basic renderer (found under `src/visuals/render.jl`).

## Setup and running
1. Clone repository `git clone https://github.com/CNCLgithub/mot` and `cd mot`
2. Switch to this branch `git checkout basic-dgp-render`
3. Run `./setup.sh cont_pull conda julia` to pull the container and setup Conda and Julia environments.
4. Enter `./run.sh julia` to get into Julia REPL
5. Enter `include("test/demo.jl")` in the Julia REPL to see how you can generate a dataset and render it using the codebase.

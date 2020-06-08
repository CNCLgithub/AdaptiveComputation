# mot-gm
Multiple object tracking generative model in Julia

## Setup and running
1. Clone repository `git clone https://github.com/CNCLgithub/mot` and `cd mot`
2. Get deps using `git submodule update --init --recursive`
2. Run `./setup.sh cont_build conda julia` to build the container and setup Conda and Julia.
3. Enter `./run.sh julia` to get into Julia REPL
4. Enter `include("scripts/inference/quick_run.jl");` in the Julia REPL to run the data generating procedure and inference using the particle filter.

using MOT
using Gen
using Gen_Compose

using JSON
using Images


function main()
    dataset = "datasets/exp_0.h5"
    trial = 0
    T = 10
    dynamics_params_path = "src/dynamics_models/brownian.json"
    inference_params_path = "src/inference/inference.json"

    open(dynamics_params_path, "r") do f
        global dynamics_params
        dynamics_params = JSON.parse(f)
    end
    
    generative_process_params = Dict("num_dots" => 1,
                                     "init_pos_spread" => 100.0,
                                     "init_vel_spread" => 10.0,
                                     "dynamics_params" => dynamics_params)

    init_positions, positions = brownian_generative_process(T, generative_process_params)

    for t=1:T
        for i=1:generative_process_params["num_dots"]
            println(i, "\t",  positions[t,i,:])
        end
        println()
    end
end


main()

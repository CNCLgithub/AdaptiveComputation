using MOT
using Gen
using Gen_Compose

using JSON
using Images
using FileIO


function main()
    dataset = "datasets/exp_0.h5"
    trial = 0
    T = 10
    dynamics_params_path = "src/dynamics_models/brownian.json"
    graphics_params_path = "src/graphics/graphics.json"
    inference_params_path = "src/inference/inference.json"
    
    dynamics_params = read_json(dynamics_params_path)
    graphics_params = read_json(graphics_params_path)
    
    generative_process_params = Dict("num_dots" => 5,
                                     "init_pos_spread" => 10.0,
                                     "init_vel_spread" => 10.0,
                                     "dynamics_params" => dynamics_params)
    
    # generating positions
    init_positions, positions = brownian_generative_process(T, generative_process_params)
    
    # rendering masks for each dot
    masks = get_masks(positions, graphics_params)
    
    for t=1:T
        for i=1:generative_process_params["num_dots"]
            mkpath("testing_masks")
            save("testing_masks/$(lpad(t,3,"0"))_$(lpad(i,3,"0")).png", masks[t,i])
            println(i, "\t",  positions[t,i,:])
        end
        println()
    end
end


main()

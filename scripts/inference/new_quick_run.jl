using MOT
using Gen
using Gen_Compose

using JSON
using Images
using FileIO


function main()
    dataset = "datasets/exp_0.h5"
    trial = 0
    T = 20

    dynamics_params_path = "src/dynamics_models/brownian.json"
    graphics_params_path = "src/graphics/graphics.json"
    inference_params_path = "src/procedures/inference.json"
    attention_params_path = "src/procedures/attention.json"
    
    dynamics_params = read_json(dynamics_params_path)
    graphics_params = read_json(graphics_params_path)
    inference_params = read_json(inference_params_path)
    
    full_params = Dict("num_dots" => 4,
                       "init_pos_spread" => 250.0,
                       "init_vel_spread" => 2.0,
                       "attended_trackers" => fill([], T), # hacky to get attention stats
                       "dynamics_params" => dynamics_params,
                       "graphics_params" => graphics_params,
                       "inference_params" => inference_params)
    
    # generating positions
    init_positions, positions = brownian_generative_process(T, full_params)
    
    # rendering masks for each dot
    masks = get_masks(positions, graphics_params)

    # running inference
    results = run_inference(masks, init_positions, full_params)
    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]

    # getting the images
    full_imgs = get_full_imgs(masks)
    
    # this is visualizing what the observations look like (and inferred state too)
    # you can find images under full_imgs/
    visualize(tracker_positions, full_imgs, full_params)
end


main()

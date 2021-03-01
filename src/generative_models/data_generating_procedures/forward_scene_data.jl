function forward_scene_data!(scene_data, timestep)
    scene_data[:gt_causal_graphs] = scene_data[:gt_causal_graphs][timestep:end]
    scene_data[:masks] = scene_data[:masks][timestep:end]
end

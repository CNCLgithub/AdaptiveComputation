using MOT
using Random
Random.seed!(1)

function render_stimuli(k,
                        pylon_strength,
                        vel,
                        first_pi,
                        second_pi,
                        name)

    motion = ISRPylonsDynamics(vel = vel,
                               pylon_strength = pylon_strength,
                               pylon_radius = 100.0,
                               pylon_x = 150.0,
                               pylon_y = 150.0)
    cm = choicemap()
    for i=1:Int(default_gm.n_trackers+default_gm.distractor_rate)
        cm[:init_state => :trackers => i => :pylon_interaction] = first_pi[i]+2
        cm[:kernel => floor(Int, k/2) => :dynamics => :pylon => i => :stay] = false
        cm[:kernel => floor(Int, k/2) => :dynamics => :pylon => i => :pylon_interaction] = second_pi[i]+2
    end
    
    scene_data = nothing
    tries = 0
    while true
        tries += 1
        print("tries: $tries \r")
        scene_data = dgp(k, default_gm, motion;
                         generate_masks=false,
                         cm=cm)
        is_min_distance_satisfied(scene_data, 80.0) && break
    end
        

    render(default_gm, k;
           gt_causal_graphs=scene_data[:gt_causal_graphs],
           path=joinpath("render", "$(name)_vel_$(vel)_force_$(pylon_strength)"),
           freeze_time=24,
           highlighted=collect(1:4),
           stimuli=true)
end


k = 240
pylon_strength = 25
vel = 12
het_pi = [-1, -1, 1, 1, -1, -1, 1, 1]
hom_att_pi = fill(1, 8)
hom_rep_pi = fill(-1, 8)

render_stimuli(k, pylon_strength, vel, het_pi, hom_att_pi, "het_homatt")
render_stimuli(k, pylon_strength, vel, het_pi, hom_rep_pi, "het_homrep")

render_stimuli(k, pylon_strength, vel, hom_att_pi, het_pi, "homatt_het")
render_stimuli(k, pylon_strength, vel, hom_rep_pi, het_pi, "homrep_het")

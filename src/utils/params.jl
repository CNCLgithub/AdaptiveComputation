export Params,
    default_params

mutable struct Params
    inertia::Float64
    spring::Float64
    sigma_w::Float64
    sigma_v::Float64

    #position_noise::Float64
    #depth_noise::Float64

    num_trackers::Int
    num_distractors_rate::Float64
    
    rejuv_smoothness::Float64 # lower = smoother
    max_rejuv::Int

    area_width::Int
    area_height::Int

    img_width::Int
    img_height::Int
    #img_noise::Float64
    dot_radius::Float64

    attended_trackers::Vector{Vector{Int}} # fake param to store attention info
end

const default_params = Params(0.8,      #intertia
                              0.002,    #spring
                              1.5,      #sigma_w
                              5.0,    #sigma_v

                              #5.00,      #position_noise
                              #0.2,      #depth noise

                              1,        #num_trackers
                              1.0,      #num_distractors_rate
                              
                              1.03,     #rejuv_smoothness
                              15,         #max_rejuv
                                
                              800,      #area_width
                              800,      #area_height

                              200,      #img_width
                              200,       #img_height
                              #0.1,      #img_noise
                              20.0,      #dot_radius

                              [] #attended trackers fake param
                             )

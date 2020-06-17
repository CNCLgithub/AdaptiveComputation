export overlay

using Luxor
# using Printf

# creating directory if it doesn't exist
if !ispath("render") mkdir("render") end

scale = 2

obs_size = 10 * scale
target_size = 5 * scale
pf_size = 1 * scale

font_size = 20 * scale


function _render_masks(obs, timestep)
	num_observations = size(obs, 1)
	setopacity(1.0)
	for o=1:num_observations
		point_ob = Luxor.Point(obs[o,1], -obs[o,2])
		sethue("black")
		Luxor.circle(point_ob, obs_size, :fill)
	end
end



function _render_observations(obs, timestep; show_label=true, 
							  highlighted::Union{Nothing,Vector{Int}} = nothing, hue="lightsalmon2")
    #println(obs)
	num_observations = size(obs, 1)
	setopacity(1.0)
    for (o,ob) in enumerate(obs)
		point_ob = Luxor.Point(ob[1], -ob[2])
		sethue(hue)
		Luxor.circle(point_ob, obs_size, :fill)
		if show_label
			sethue("black")
			Luxor.text(string(o), point_ob, halign=:right, valign=:bottom)
		end
	end

	if typeof(highlighted) != Nothing
		for t in highlighted
            point_ob = Luxor.Point(obs[t][1], -obs[t][2])
			sethue("red")
			Luxor.circle(point_ob, target_size, :fill)
		end
	end
end

function _render_pf(pf_xy, timestep; color="darkslateblue", assignments=nothing, attended=nothing)
	(_, num_particles, num_targets, _) = size(pf_xy)

	#opc = 10.0/num_particles
	opc = 1.0

	for t=1:num_targets
		mu_pos = mean(pf_xy[timestep,:,t,:], dims = 1)
		std_pos = std(pf_xy[timestep,:,t,:], dims = 1)
		
		mu_pos[2] *= -1
		setopacity(opc)
		#Luxor.ellipse(mu_pos..., std_pos..., :stroke)
		for p=1:num_particles
			if !isnothing(assignments)
				sethue(assignments[timestep,p,t] <= num_targets ? "blue" : "red" )
			end
			point_pred = Luxor.Point(pf_xy[timestep,p,t,1], -pf_xy[timestep,p,t,2])
            if !isnothing(attended)
                bg_visibility = 1.0 - p*(1.0/num_particles)
                prev_bg_visibility = 1.0 - (p-1)*(1.0/num_particles)
                opacity = 1.0 - bg_visibility/prev_bg_visibility
                opacity *= attended[timestep,t]

                setopacity(opacity)
                sethue("red")
			    Luxor.circle(point_pred, pf_size*3, :fill)
                     
                # setting things back
                setopacity(1.0)
            end
		    sethue(color)
			#setopacity(opc)
			Luxor.circle(point_pred, pf_size, :fill)
			Luxor.text(string(t), point_pred, halign=:left, valign=:bottom)
			#=
			if p == num_particles
				sethue("blue")
				setopacity(1.0)
				Luxor.text(string(t), point_pred, halign=:left, valign=:top)
			end
			=#
		end
	end
end

function _init_drawing(number; background_color="ghostwhite", folder_name="", obs_id="")
        # obs_id only used in rendering the mask
	Drawing(800, 800, "render/"*folder_name*lpad(number, 3, "0")*obs_id*".png")
	origin()
	background(background_color)
	Luxor.fontsize(font_size)
    point = Luxor.Point(300, 380)
    sethue("black")
    Luxor.text(string(number), point)
end


# renders the observations and optionally the particle filter infered positions
function overlay(optics, num_targets; pf_xy=nothing, pf_dxy=nothing,
                 folder_name="", run=nothing,
				 stimuli=false, freeze_time=50,
                 highlighted=nothing, render_masks=0,
                assignments=nothing, attended=nothing)
	println("rendering the images...")
	
	# if creating stimuli, then should stop before and after the dots moving
	time_stopped = stimuli ? freeze_time : 0

	# creating specific directory if specified
	if folder_name != ""
		if !ispath("render/"*folder_name) mkdir("render/"*folder_name) end
		folder_name *= "/"
		if typeof(run) != Nothing
			folder_name *= "run_"*string(run)*"/"
			if !ispath("render/"*folder_name) mkdir("render/"*folder_name) end
		end
	end
	

    T = length(optics)
        
	# stopped at beginning
	for timestep=1:time_stopped
        obs = optics[1]
        N = size(obs, 1)

		_init_drawing(timestep, folder_name=folder_name)

		# rendering observations
		if stimuli
			_render_observations(obs, 1; show_label=false, highlighted=collect(1:num_targets))
		else
			_render_observations(obs, 1)
		end

		if typeof(pf_xy) == Nothing
			finish()
			continue
		end

		# rendering xs and ys from the particles
		if !stimuli
			_render_pf(pf_xy, 1)
		end

		finish()
	end

	for timestep=1:T
        obs = optics[timestep]
        _init_drawing(timestep+time_stopped+render_masks, folder_name=folder_name)

		# rendering observations with numbers if it is not for stimuli
		if stimuli
			_render_observations(obs, timestep; show_label=false)
		else
			_render_observations(obs, timestep; show_label=true)
		end
		
		# if there are no particle estimates then save to png and continue
		if typeof(pf_xy) == Nothing
			finish()
		end

                if render_masks > 0
                    for i=1:N
                        _init_drawing(timestep+time_stopped+render_masks, background_color="white", obs_id="_$(i)", folder_name=folder_name)
                        _render_observations(reshape(obs[i,:], (T, 1, 2)), timestep; show_label=false, hue="black")
                        finish()
                    end
                end

		# if there are no particle estimates then save to png and continue
		if typeof(pf_xy) == Nothing
			continue
		end


		# rendering xs and ys from the particles
		if !stimuli
            _render_pf(pf_xy, timestep, color="blue", assignments=assignments, attended=attended)
		end
	
		finish()
	end

	for timestep=T+1:T+time_stopped
        _init_drawing(timestep+time_stopped, folder_name=folder_name)

        obs = optics[T]

		# rendering observations
		if stimuli
			_render_observations(obs, T; show_label=false, highlighted=highlighted)
		else
			_render_observations(obs, T)
		end
		
		# if there are no particle estimates then save to png and continue
		if typeof(pf_xy) == Nothing
			finish()
			continue
		end

		# rendering xs and ys from the particles
		if !stimuli
			_render_pf(pf_xy, T)
		end

		finish()
	end
end

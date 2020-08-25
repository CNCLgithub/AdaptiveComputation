export visualize_inference

function visualize_inference(results, positions, gm, attention, path)
    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]
    k = size(tracker_positions, 1)
    # tracker_masks = get_masks(tracker_positions)
    aux_state = extracted["aux_state"]
    attention_weights = [aux_state[t].stats for t = 1:k]
    attention_weights = collect(hcat(attention_weights...)')

    out = dirname(path)
    plot_compute_weights(attention_weights, out)

    attempts = Vector{Int}(undef, k)
    attended = Vector{Vector{Float64}}(undef, k)
    for t=1:k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end
    MOT.plot_attention(attended, attention.sweeps, out)
    plot_rejuvenation(attempts, out)

    # visualizing inference on stimuli
    render(gm;
           dot_positions = positions[1:k],
           path = joinpath(out, "render"),
           pf_xy=tracker_positions[:,:,:,1:2],
           attended=attended/attention.sweeps,)
   
    # also rendering from the perspective of the model
    tracker_masks = extracted["unweighted"][:tracker_masks]
    println()
    println(size(tracker_masks))
    println()
    masks = []
    for i=1:k
        push!(masks,[])
        for j=1:gm.n_trackers
            push!(masks[i], mask(tracker_masks[i,1,j]))
        end
    end
    # masks = get_masks(positions, gm.dot_radius,
                      # gm.img_height, gm.img_width,
                      # gm.area_height, gm.area_width)
    full_imgs = get_full_imgs(masks)
    visualize(tracker_positions, full_imgs, gm, path=joinpath(out, "model_render"))
end


function visualize(xy, full_imgs, params; path="model_render")
    k, n, _, _ = size(xy)

    h, w, ah, aw = params.img_height, params.img_width, params.area_height, params.area_width

    for t=1:k
        img = full_imgs[t]

        for p=1:n
            for i=1:size(xy,3)
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x, y = translate_area_to_img(x, y, h, w, ah, aw)

                # particle positions
                draw_circle!(img, [x,y], 3.0, false)
                draw_circle!(img, [x,y], 2.0, true)
                draw_circle!(img, [x,y], 1.0, false)
            end
        end

        mkpath(path)
        filename = "$(lpad(t, 3, "0")).png"
        save(joinpath(path, filename), img)
    end
end

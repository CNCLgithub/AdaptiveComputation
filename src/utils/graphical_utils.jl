export add_images,
       subtract_images,
       translate_area_to_img,
       draw_circle!

using Images



# drawing a circle 
function draw_circle!(img, center, radius, value)
    for i=1:size(img,1)
        for j=1:size(img,2)
            if dist(center, [i,j]) < radius
                img[j,i] = value
            end
        end
    end
end


function draw_gaussian_mask(object, img_so_far, params)
    x, y = translate_area_to_img(object[1], object[2], params)
    mask = zeros(params.img_height, params.img_width)

    radius = params.dot_radius * params.img_width / params.area_width

    mask = draw_gaussian_circle(mask, [x,y], radius/2)
    
    #println(sort(mask[:],rev=true)[1:5])
    return mask
end


# add images and clamp to normal range
function add_images(img1, img2)
    img = map(clamp01nan, img1+img2)
    img .-= 1e-10
    return img
end

# subtract images and clamp to normal range
function subtract_images(img1, img2)
    img = map(clamp01nan, img1-img2)
    img .+= 1e-10
    return img
end

# draws a full image from masks of one timestep
function get_full_img(masks)
    img_height, img_width = size(first(masks))
    img = BitArray{2}(undef, img_height, img_width)
    img .= false

    for mask in collect(masks)
        img = img .|= mask
    end

    return img
end

# get a full sequency of masks added together on images
function get_full_imgs(masks)
    T = size(masks, 1)
    full_imgs = []
    for t=1:T
        img = get_full_img(masks[t,:])
        push!(full_imgs, img)
    end
    return full_imgs
end


function visualize(xy, full_imgs, params, folder="visuals")
    mkpath(folder)

    T = length(full_imgs)
    num_particles = params["inference_params"]["num_particles"]

    for t=1:T
        img = full_imgs[t]

        for p=1:num_particles
            for i=1:size(xy,3)
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x, y = translate_area_to_img(x, y, params["graphics_params"])

                draw_circle!(img, [x,y], 5.0, false)
                draw_circle!(img, [x,y], 3.0, true)
                draw_circle!(img, [x,y], 1.0, false)
            end
        end
        
        fn = "$(lpad(t, 3, "0")).png"
        save(joinpath(folder, fn), img)
    end

end

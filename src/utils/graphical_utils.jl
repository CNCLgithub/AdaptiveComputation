export add_images,
       subtract_images,
       translate_area_to_img,
       draw_circle!,
       draw_gaussian_circle,
       draw_mask,
       draw_gaussian_mask

using Images


# 2d gaussian function
function two_dimensional_gaussian(x, y, x_0, y_0, A, sigma_x, sigma_y)
    return A * exp(-( (x-x_0)^2/(2*sigma_x^2) + (y-y_0)^2/(2*sigma_y^2)))
end

# drawing a gaussian circle 
using Distributions
function draw_gaussian_circle(img, center, radius)
    # standard deviation based on the volume of the Gaussian
    spread_1 = 1.0 # parameter for how spread out the mask is
    spread_2 = 5.0
    A = 0.4999999999
    std_1 = sqrt(spread_1 * radius)
    std_2 = sqrt(spread_2 * radius)

    #A = 0.5/spread^2
    
    for i=1:size(img,1)
        for j=1:size(img,2)
            img[j,i] = two_dimensional_gaussian(i, j, center[1], center[2], A, std_1, std_1) 
            img[j,i] += two_dimensional_gaussian(i, j, center[1], center[2], A, std_2, std_2) 
        end
    end
    
    return img
end

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


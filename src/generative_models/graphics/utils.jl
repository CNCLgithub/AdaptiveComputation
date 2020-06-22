export add_images,
       subtract_images,
       translate_area_to_img,
       draw_circle!

using Images

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
        img = get_full_img(masks[t])
        push!(full_imgs, img)
    end
    return full_imgs
end


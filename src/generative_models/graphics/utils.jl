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

# get a full sequency of masks added together on images
function get_full_imgs(masks)
    k = length(masks)

    full_imgs = Vector{BitArray{2}}(undef, k)
    for t=1:k
        # convert each mask to Array{Bool} and do an OR operation over all of them
        full_imgs[t] = mapreduce(x->convert(Array{Bool}, x), (x,y) -> x.|y, masks[t])
    end
    return full_imgs
end


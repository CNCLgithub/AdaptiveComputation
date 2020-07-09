using PyCall
using Images, Colors

function get_uint8_vec(rgb::RGB{Normed{UInt8,8}})
    return nothing
end

mask_rcnn = PyNULL()
copy!(mask_rcnn, pyimport("mask_rcnn.get_masks"))

img = load("output/datasets/mask_rcnn_exp0/input_pngs/001_001.png")
mat = channelview(img)
@show size(mat)
@show mat[:,1,1]

masks = mask_rcnn.get_masks(mat)
for i=1:size(masks,1)
    mask = masks[i,:,:]
    save("testing_mask_$i.png", mask)
end

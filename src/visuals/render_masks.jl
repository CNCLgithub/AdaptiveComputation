export render_masks

#using PaddedViews
using ImageTransformations

# this takes masks within one receptive field and superimposes them
aggregate_masks(masks::Vector{Matrix{Float64}}) = sum(masks)
_or = (x, y) -> x .| y
aggregate_masks(masks::Vector{BitMatrix}) = reduce(_or, masks)

# takes already aggregated_masks for each receptive_field and
# also a vector of receptive_fields and returns the full composed
# img to save
function get_img(aggregated_masks::Vector{Matrix{Float64}})::Matrix{Float64}
    img = zeros(first(aggregated_masks))
    @inbounds for i = 1:length(aggregated_masks)
        img = maximum.(img, aggregated_masks[i])
    end
    return img
end

function get_masks(cgs::Vector{CausalGraph}, t::Int64)::Vector{Matrix{Float64}}
    vs = get_object_verts(cgs[t], Dot)
    @>> vs begin
        map(v -> get_prop(cgs[t], v, :space))
    end
end

function get_masks(choices::ChoiceMap, t::Int64)
    masks = choices[:kernel => t => :masks]
    collect(BitMatrix, masks)
end

function render_masks(data::Union{Vector{CausalGraph}, ChoiceMap}, t::Int64,
                         gm::AbstractGMParams,
                         graphics::AbstractGraphics)
    masks = get_masks(data, t)
    img = aggregate_masks(masks)
    # img = get_img(aggregated)
    imresize(img, ratio=gm.area_width/size(img, 1))
end
function render_masks(data::Union{Vector{CausalGraph}, ChoiceMap}, t::Int64,
                         gm::AbstractGMParams,
                         graphics::AbstractGraphics,
                         out_dir::String)
    img = render_masks(data, t, gm, graphics)
    path = joinpath(out_dir, "$t.png")
    save(path, img)
end

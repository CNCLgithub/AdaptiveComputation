export render_rf_masks

#using PaddedViews
using ImageTransformations

# this takes masks within one receptive field and superimposes them
_aggregate_masks(rf_masks::Vector{Matrix{Float64}}) = sum(rf_masks)
_or = (x, y) -> x .| y
_aggregate_masks(rf_masks::Vector{BitArray}) = reduce(_or, rf_masks)

# takes already aggregated_masks for each receptive_field and
# also a vector of receptive_fields and returns the full composed
# img to save
function get_img(aggregated_masks::Vector{Matrix{Float64}},
                 receptive_fields::Vector{RectangleReceptiveField})::Matrix{Float64}
    # we take the last receptive_field and take the second point
    # to find the full dimensions of the image
    (dim2, dim1) = receptive_fields[end].p2
    img = zeros(dim1, dim2)
    
    for (i, rf) in enumerate(receptive_fields)
        img[rf.p1[2], rf.p1[1]:rf.p2[1]] .= 0.5
        img[rf.p2[2], rf.p1[1]:rf.p2[1]] .= 0.5
        img[rf.p1[2]:rf.p2[2], rf.p1[1]] .= 0.5
        img[rf.p1[2]:rf.p2[2], rf.p2[1]] .= 0.5
    end

    for (i, rf) in enumerate(receptive_fields)
        img[rf.p1[2]:rf.p2[2], rf.p1[1]:rf.p2[1]] += aggregated_masks[i]
    end
    
    clamp!(img, 0.0, 1.0)

    return img
end

function get_rf_masks(cgs::Vector{CausalGraph}, t::Int64, rf::Int64,
                      receptive_fields::Vector{RectangleReceptiveField})::Vector{Matrix{Float64}}
    # init with zero mask
    #@unpack p1, p2 = receptive_fields[rf]
    #init_rf_mask = zeros((p2 .- p1 .+ (1,1))...)
    init_rf_mask = zeros(get_dimensions(receptive_fields[rf])...)

    vs = get_prop(cgs[t], :graphics_vs)
    rf_masks = @>> vs begin
        map(v -> get_prop(cgs[t], v, :space))
        cropfilter(receptive_fields[rf])
    end
    push!(rf_masks, init_rf_mask)
    return rf_masks
end

function get_rf_masks(choices::ChoiceMap, t::Int64, rf::Int64,
                      receptive_fields::Vector{RectangleReceptiveField})::Vector{BitArray}
    # init with zero mask
    # @unpack p1, p2 = receptive_fields[rf]
    # dims = p2.-p1 .+ (1,1)
    # init_rf_mask = zeros(dims...)
    init_rf_mask = zeros(get_dimensions(receptive_fields[rf])...)
    init_rf_mask = convert(BitArray, init_rf_mask)

    rf_masks = choices[:kernel => t => :receptive_fields => rf => :masks]
    push!(rf_masks, init_rf_mask)
    return rf_masks
end

function render_rf_masks(data::Union{Vector{CausalGraph}, ChoiceMap}, t::Int64,
                         gm::AbstractGMParams,
                         graphics::AbstractGraphics,
                         out_dir::String)
    
    masks = @>> 1:length(graphics.receptive_fields) begin
        map(rf -> get_rf_masks(data, t, rf, graphics.receptive_fields))
    end

    aggregated_masks = @>> masks begin
        map(rf_masks -> _aggregate_masks(rf_masks))
        collect(Matrix{Float64})
    end
    img = get_img(aggregated_masks, graphics.receptive_fields)

    #img = imresize(img, ratio=gm.area_width/size(img, 1))
    path = joinpath(out_dir, "$t.png")
    save(path, img)
end

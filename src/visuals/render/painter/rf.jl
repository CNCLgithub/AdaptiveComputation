export RFPainter

@with_kw struct RFPainter <: Painter
    area_dims::Tuple{Int64, Int64}
    rf_dims::Tuple{Int64, Int64}
    opacity::Float64 = 0.03
    # overlap::Float64 =
end


function paint(p::RFPainter, cg::CausalGraph)
    @unpack area_dims, rf_dims, opacity = p
    sethue("black")
    tiles = Tiler(area_dims..., rf_dims..., margin=0)
    foreach(tile -> box(tile[1], tiles.tilewidth, tiles.tileheight, :stroke), tiles)
    # setopacity(0.03)
    # setline(receptive_fields_overlap/gm.img_width*gm.area_width*2)
    # foreach(tile -> box(tile[1], tiles.tilewidth, tiles.tileheight, :stroke), tiles)
end

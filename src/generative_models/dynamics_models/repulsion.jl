function distance_to(a::Object, b::Object)
    pos(a) - pos(b)
end

# TODO @eivinas implement this
function distance_to(a::Wall, b::Object)
    findfirst(a)
end


function distance_to(a::Wall, b::Polygon)
    # TODO
end

function repulsion(dm::AbstractDynamicsModel, a::Object, b::Object)
    error("not implemented")
end

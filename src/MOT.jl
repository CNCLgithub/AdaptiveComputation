module MOT

using Gen
using JSON
using Parameters: @with_kw

function __init__()
    @load_generated_functions
end

include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("distributions/distributions.jl")
include("visuals/visuals.jl")

end

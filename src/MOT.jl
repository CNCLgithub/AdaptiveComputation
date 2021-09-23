module MOT

using Gen
using Gen_Compose
using GenRFS
using FillArrays
using SparseArrays
using Parameters: @with_kw, @unpack, @pack!
using Lazy: @>, @>>


# function __init__()
# end

include("world/world.jl")
include("distributions/distributions.jl")
include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("inference/inference.jl")
include("visuals/visuals.jl")

@load_generated_functions
end

module MOT

using Gen
using Gen_Compose
using GenRFS
using FillArrays
using SparseArrays
using Parameters: @with_kw, @unpack
using Lazy: @>, @>>


function __init__()
    @load_generated_functions
end

include("world/world.jl")
include("distributions/distributions.jl")
include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("procedures/procedures.jl")
include("visuals/visuals.jl")
include("experiments.jl")

end

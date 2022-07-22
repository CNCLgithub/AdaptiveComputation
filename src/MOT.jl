module MOT

using Gen
using Gen_Compose
using GenRFS
using FillArrays
using SparseArrays
using StaticArrays
using Accessors: setproperties
using Parameters: @with_kw, @unpack, @pack!
using Lazy: @>, @>>

include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("inference/inference.jl")
include("visuals/visuals.jl")

@load_generated_functions

end

module MOT

using Gen
using Gen_Compose
using GenRFS
using LinearAlgebra
using FillArrays
using SparseArrays
using StaticArrays
using Lazy: @>, @>>
using Accessors: setproperties, @set
using Parameters: @with_kw, @unpack, @pack!

include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("inference/inference.jl")
@load_generated_functions

include("visuals/visuals.jl")


end

module MOT

using Gen
using Gen_Compose
using GenRFS
using Parameters: @with_kw
using PyCall

mask_rcnn = PyNULL()

function __init__()
	#copy!(npp_lib, pyimport("npp.permutation_sample"))
	copy!(mask_rcnn, pyimport("mask_rcnn.get_masks"))

    # setup gen static functions
    #Gen.load_generated_functions()
    @load_generated_functions
end

include("utils/utils.jl")
include("generative_models/generative_models.jl")
include("distributions/distributions.jl")
include("procedures/procedures.jl")
include("visuals/visuals.jl")
include("experiments/experiments.jl")

end

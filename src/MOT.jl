module MOT

using Gen
using Gen_Compose
using GenRFS

function __init__()
    # setup gen static functions
    Gen.load_generated_functions()

    # import pycall libraries here
	# e.g. copy!(npp_lib, pyimport("npp.permutation_sample"))
    # ...
end

# fundamentals
include("utils/utils.jl")
include("generative_processes/generative_processes.jl")
include("generative_models/generative_models.jl")
include("distributions/distributions.jl")
include("procedures/procedures.jl")

# extra
include("plotting/plotting.jl")
include("visuals/visuals.jl")

end

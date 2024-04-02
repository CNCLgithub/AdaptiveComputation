include("gpp.jl")
include("von_mises.jl")
include("id.jl")
include("log_bern_element.jl")
include("iso_element.jl")

"Declaration of approximate random finite set for motion observations"
const gpp_mrfs = MRFS{GaussObs{2}}()

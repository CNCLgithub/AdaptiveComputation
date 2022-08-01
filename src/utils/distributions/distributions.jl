include("mask.jl")
include("von_mises.jl")
include("id.jl")
include("log_bern_element.jl")

const mask_rfs = RFS{BitMatrix}()
const mask_mrfs = MRFS{Matrix{Bool}}()

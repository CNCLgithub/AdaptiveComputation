export softmax, normalize_weights



# Numerical constants
const ln_hlf = log(0.5)
const two_pi_sqr = 4.0 * pi * pi

log1mexp(a::Float64) = (a < ln_hlf) ? log1p(-exp(a)) : log(-expm1(a))

# stable softmax
# function softmax(x::Array{Float64}; t = 1.0)
#     max_x = maximum(x)
#     if max_x === -Inf
#         nx = length(x)
#         unit = 1.0 / nx
#         return fill(unit, nx)
#     end
#     x = x .- maximum(x)
#     exp.(x) / logsumexp(x)
# end
function softmax(x::Array{Float64}; t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x; t = t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}; t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

# based of numpys implementation
# https://github.com/numpy/numpy/blob/c900978d5e572d96ccacaa97af28e2c5f4a0b137/numpy/core/src/npymath/npy_math_internal.h.src#L642
function logaddexp(a::Real, b::Real)
    if (a == b)
        return a + log(2)
    else
        tmp = a - b
        if tmp > 0
            return a + log1p(exp(-tmp))
        else
            return b + log1p(exp(tmp))
        end
    end
end


function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

# computes distance between two points
function dist(x,y)
    return sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
end

function clamp_and_round(v::Float64, c::Int64)::Int64
    @> v begin
        clamp(1., c)
        (@>> round(Int64))
    end
end


# translates coordinate from euclidean to image space
function translate_area_to_img(x::Float64, y::Float64,
                               img_width::Int64, img_height::Int64,
                               area_width::Float64, area_height::Float64)

    x = x * img_width/area_width
    x += img_width/2

    # inverting y
    y = y * -1 * img_height/area_height
    y += img_height/2

    return x, y
end

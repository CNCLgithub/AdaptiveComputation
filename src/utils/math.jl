export softmax, normalize_weights



# Numerical constants
const ln_hlf = log(0.5)
const two_pi_sqr = 4.0 * pi * pi

# stable softmax
function softmax(x::Array{Float64})
    max_x = maximum(x)
    if max_x === -Inf
        nx = length(x)
        unit = 1.0 / nx
        return fill(unit, nx)
    end
    x = x .- maximum(x)
    exp.(x) / sum(exp.(x))
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


# stable softmax
function softmax(x)
    x = x .- maximum(x)
    return exp.(x) / sum(exp.(x))
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

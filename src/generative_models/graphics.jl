
################################################################################
# Helpers
################################################################################



"""
Writes out a dot mask to matrix
"""
function exp_dot_mask(
                       x0::Float64, y0::Float64,
                       r::Float64,
                       w::Int64, h::Int64,
                       outer_f::Float64,
                       inner_f::Float64,
                       outer_p::Float64,
                       inner_p::Float64)

    outer_r = r  * outer_f
    inner_r = r  * inner_f

    # half-life is 1/6 outer - inner
    hl = 6.0 * ln_hlf / abs(outer_r - inner_r)

    xlow = clamp_and_round(x0 - outer_r, w)
    xhigh = clamp_and_round(x0 + outer_r, w)
    ylow = clamp_and_round(y0 - outer_r, h)
    yhigh = clamp_and_round(y0 + outer_r, h)
    n = (xhigh - xlow + 1) * (yhigh - ylow + 1)
    Is = zeros(Int64, n)
    Js = zeros(Int64, n)
    Vs = zeros(Float64, n)
    k = 0
    @inbounds @fastmath for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        k +=1
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        # flip i and j in mask
        Is[k] = j
        Js[k] = i
        (dst > outer_r) && continue
        Vs[k] = (dst <= inner_r ) ? inner_p : outer_p * exp(hl * dst)
    end
    sparse(Is, Js, Vs, h, w)
end

function exp_dot_mask!(m,
                       x0::Float64, y0::Float64,
                       r::Float64,
                       w::Int64, h::Int64,
                       outer_f::Float64,
                       inner_f::Float64,
                       outer_p::Float64,
                       inner_p::Float64)

    outer_r = r  * outer_f
    inner_r = r  * inner_f

    # half-life is 1/6 outer - inner
    hl = 3.0 * ln_hlf / abs(outer_r - inner_r)

    xlow = clamp_and_round(x0 - outer_r, w)
    xhigh = clamp_and_round(x0 + outer_r, w)
    ylow = clamp_and_round(y0 - outer_r, h)
    yhigh = clamp_and_round(y0 + outer_r, h)
    k = 0
    for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        k +=1
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        (dst > outer_r) && continue
        v = (dst <= inner_r ) ? inner_p : outer_p * exp(hl * dst)
        v < 1E-4 && continue
        # flip i and j in mask
        mv = m[j, i]
        if (iszero(mv) || mv < v)
            m[j, i] = v
        end
    end
    return nothing
end

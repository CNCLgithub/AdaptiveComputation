using Gen

a = Array{Float64, 3}(undef, 1, 1, 2)
b = Array{Float64, 3}(undef, 1, 2, 2)

idx = npp(a, b)
w = Gen.logpdf(npp, idx, a, b)
println(idx, w)

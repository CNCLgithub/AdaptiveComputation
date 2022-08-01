using Gen
using MOT
using Profile
using SparseArrays
using BenchmarkTools
using StatProfilerHTML


function main()
    ws[xs] .= 0.5
    sws = sparse(ws)
    Profile.init(delay = 1E-5,
                 n = 10^6)
    Profile.clear()
    Gen.logpdf(mask, xs, sws)
    display(@benchmark Gen.logpdf(mask, $xs, $sws))
    @profilehtml for _ = 1:1000
        Gen.logpdf(mask, xs, sws)
    end
end

# main();

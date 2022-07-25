using Gen
using MOT
using Profile
using SparseArrays
using StatProfilerHTML

function main()
    ws = zeros((100, 100))
    xs = rand(100, 100) .> 0.80
    ws[xs] .= 0.5
    sws = sparse(ws)
    Profile.init(delay = 1E-8,
                 n = 10^8)
    Profile.clear()
    Gen.logpdf(mask, xs, sws)
    display(@benchmark Gen.logpdf(mask, $xs, $sws))
    @profilehtml for _ = 1:1000
        Gen.logpdf(mask, xs, sws)
    end
end

main();

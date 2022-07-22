using Gen
using MOT
using Profile
using StatProfilerHTML

function main()
    ws = fill(0.5, 200, 200)
    xs = rand(200, 200) .> 0.5



    Profile.init(delay = 1E-5,
                 n = 10^6)
    @show typeof(ws)
    @show typeof(xs)
    @profilehtml Gen.logpdf(mask, xs, ws)
    @time Gen.logpdf(mask, xs, ws)
    @profilehtml Gen.logpdf(mask, xs, ws)
end

main();

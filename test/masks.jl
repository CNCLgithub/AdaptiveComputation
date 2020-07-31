using GenRFS
using Gen
using Profile
using ProfileView
using StatProfilerHTML
using PProf

positions = [[-100., -100, 1],
             [-100., 100, 2],
             [100., -100, 3],
             [100., 100, 4]]
positions = hcat(positions...)'
gm_params = MOT.GMMaskParams(img_height = 50,
                             img_width = 50)

mask_args = first(MOT.get_masks_rvs_args(positions, gm_params))

pmbrfs = RFSElements{Array}(undef, 5)
pmask = fill(0.1, gm_params.img_height, gm_params.img_width)
pmbrfs[1] = PoissonElement{Array}(4, mask, (pmask,))
r = 0.95
for i = 2:5
    pmbrfs[i] = BernoulliElement{Array}(r, mask, mask_args[i-1])
end

xs = fill(mask(first(mask_args)...), 8)
logpdf(rfs, xs, pmbrfs)
Profile.clear()
Profile.init(;n = 1000000, delay = 1E-5)
@time logpdf(rfs, xs, pmbrfs)
@profilehtml logpdf(rfs, xs, pmbrfs)
# @profile logpdf(rfs, xs, pmbrfs)
# pprof()

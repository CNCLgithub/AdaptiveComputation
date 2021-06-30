using MOT
using GenRFS
using Gen
#using Profile
#using ProfileView
#using StatProfilerHTML
#using PProf

dot = Dot(pos = zeros(3),
          vel = zeros(2))
gm = MOT.load(GMParams, "/project/scripts/inference/exp1_difficulty/gm.json")
pmbrfs, _ = MOT.get_masks_params([dot.pos], gm, flow_masks=nothing)
xs = rfs(pmbrfs)
logpdf(rfs, xs, pmbrfs)

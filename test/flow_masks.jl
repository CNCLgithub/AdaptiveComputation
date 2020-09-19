using MOT
using Random
Random.seed!(2)

k = 120
gm = GMMaskParams(fmasks=true,
                  fmasks_decay_function=x->x/2,
                  distractor_rate=0)
motion = BrownianDynamicsModel()

scene_data = dgp_gm(k, gm, motion)
full_imgs = MOT.get_full_imgs(scene_data[:masks])
MOT.model_render(full_imgs, gm)


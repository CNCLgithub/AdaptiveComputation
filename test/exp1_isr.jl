using MOT

q = Exp1(k=20,
         scene=1,
         dataset_path="/datasets/exp1_isr.jld2",
         gm="/project/scripts/inference/exp1_isr/gm.json",
         proc="/project/scripts/inference/exp1_isr/proc.json")

att = MOT.load(MapSensitivity, "/project/scripts/inference/exp0/td.json",
               sweeps=0)

path = "/experiments/test/test.jld2"

run_inference(q, att, path, viz=true)

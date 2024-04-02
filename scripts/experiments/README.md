# Experiments

Each directory contains the scripts to configure and run the adaptive computation model for the respective experiment.

The model is configured by the following json files:

- `gm.json`: Parameters for the world model (eg., prior over object speeds)
- `td.json`: The attention parameters (ie., for arousal and importance)
- `proc.json`: The parameters for the particle filter

Each directory contains a main script to run the model on a given trial (the datasets that contain these trials are found under `env.d/spaths/datasets`).
For example, the probes experiment has a main script under `scripts/experiments/probes/exp_probes.jl`.
We also provide scripts to batch the main script via SLURM (e.g., `scripts/experiments/probes/batch.py`). Note that this batch scripts may require additional configuration depending on the cluster environment. 

The results of these scripts are placed under `env.d/spaths/experiments`

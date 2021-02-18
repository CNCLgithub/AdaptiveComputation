#!/bin/bash

. load_config.sh

usage="$(basename "$0") [targets...] -- setup an environmental component of the project according to [default|local].conf
supported targets:
    cont_[pull|build] : either pull the singularity container or build from scratch
    conda : build the conda environment
    julia : build julia environment
    datasets : pull datasets
    checkpoints : pull checkpoints (NN weights)
"

[ $# -eq 0 ] || [[ "${@}" =~ "help" ]] && echo "$usage"

# container setup
[[ "${@}" =~ "cont_build" ]] || [[ "${@}" =~ "cont_pull" ]] || echo "Not touching container"
[[ "${@}" =~ "cont_pull" ]] && echo "pulling container" && \
    wget "https://yale.box.com/shared/static/i5vxp5xghtfb2931fhd4b0ih4ya62o2s.sif" \
    -O "${ENV[cont]}"
[[ "${@}" =~ "cont_build" ]] && echo "building container" && \
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity


# conda setup
[[ "${@}" =~ "conda" ]] || echo "Not touching conda"
[[ "${@}" =~ "conda" ]] && echo "building conda env" && \
    singularity exec ${ENV[cont]} bash -c "yes | conda create -p $PWD/${ENV[env]} python=3.6" && \
    ./run.sh python -m pip install -r requirements.txt && \
    # detectron2 setup (dependent on torch, so out of the requirements.txt
    # since pip installs in the alphabetical order)
    ./run.sh python -m pip install -e deps/detectron2

# julia setup
[[ "${@}" =~ "julia" ]] || echo "Not touching julia"
[[ "${@}" =~ "julia" ]] && echo "building julia env" && \
    ./run.sh julia -e '"using Pkg; Pkg.instantiate()"'

# datasets
[[ "${@}" =~ "datasets" ]] || [[ "${@}" =~ "datasets" ]] || echo "Not touching datasets"
[[ "${@}" =~ "datasets" ]] && echo "pulling datasets" && \
    mkdir -p output/datasets && \
    echo "pulling exp_0 dataset" && \
    wget "https://yale.box.com/shared/static/2kt5psxh7nyb5s3s09g4kwhxnjmigcjs.h5" \
        -O "output/datasets/exp_0.h5" && \
    wget "https://yale.box.com/shared/static/0jeyo01zkm4viwmqsm9qzdy3uatbd755.jld2" \
        -O "output/datasets/exp0.jld2" && \
    wget "https://yale.box.com/shared/static/ikq1e26l8xbhrac8x3cxwei972g0judo.jld2" \
        -O "output/datasets/exp1.jld2" && \
    wget "https://yale.box.com/shared/static/75accvgspil2sr34tg2g3e37uvhjkxz6.jld2" \
        -O "output/datasets/exp1_isr.jld2" && \
    wget "https://yale.box.com/shared/static/wwpwkh7ara5wqmr8bde3jvvdael8x783.jld2" \
        -O "output/datasets/exp1_brownian.jld2"  && \
    wget "https://yale.box.com/shared/static/f7w0a43ubp8akiknemp77da1xg3gr64h.jld2" \
        -O "output/datasets/exp1_isr_480.jld2"

# checkpoints
[[ "${@}" =~ "checkpoints" ]] || [[ "${@}" =~ "checkpoints" ]] || echo "Not touching checkpoints"
[[ "${@}" =~ "checkpoints" ]] && echo "pulling checkpoints" && \
    mkdir -p checkpoints && \
    echo "pulling mask_rcnn_weights_0" && \
    wget "https://yale.box.com/shared/static/85eiwmnthuy93bqrpwb5hwdo70qndmbk.pth" \
    -O "output/checkpoints/mask_rcnn_weights_0.pth"

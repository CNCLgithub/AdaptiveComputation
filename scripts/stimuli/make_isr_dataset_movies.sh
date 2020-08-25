#!/bin/bash

n_trials=20
dataset_path="output/renders/isr_dataset/test/"
mkdir "$dataset_path/videos"


for i in {1..20}
do
    path="$dataset_path/$i/%3d.png"
    echo $path
    ffmpeg -framerate 24 -i $path "$dataset_path/videos/$i.mp4"
done

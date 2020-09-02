#!/bin/bash

n_trials=2
dataset_path="output/renders/brownian_dataset"
mkdir "$dataset_path/videos"

for ((i=1;i<=n_trials;i++))
do
    ffmpeg -framerate 24 -i "$dataset_path/$i/%3d.png" "$dataset_path/videos/$i.mp4"
done

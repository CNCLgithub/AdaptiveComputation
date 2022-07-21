#!/usr/bin/env python

import os
import json
import argparse
import pandas as pd

angle = 0 # not using rotation

def generate_condlist(data, outpath:str):

    scenes = data.scene.unique().tolist()
    condlist = []
    # just one condition
    for cond in range(1):
        cond_trials = []

        for (i, scene) in enumerate(scenes):
            # mapping tracker_trial to the scene's trackers
            scene_data = data[data.scene == scene]
            frames = scene_data.frame.unique()
            scene_probes = []
            for (j, frame) in enumerate(frames):
                tracker = scene_data[scene_data.frame == frame]['tracker']
                scene_probes.append([int(tracker), int(frame)])

            cond_trials.append([scene, angle, scene_probes])

        condlist.append(cond_trials)

    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    data = pd.read_csv('/spaths/datasets/exp2_probes_hatt_timings.csv')
    generate_condlist(data, '/spaths/datasets/exp2_probes_hatt_condlist.json')

if __name__ == '__main__':
    main()

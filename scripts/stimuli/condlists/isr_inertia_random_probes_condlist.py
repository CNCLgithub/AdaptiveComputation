#!/usr/bin/env python

import os
import json
import argparse
import pandas as pd
import numpy as np

def generate_condlist(scenes, data, outpath):
    
    condlist = []

    # just one condition for now
    for cond in range(1):
        cond_trials = []
        
        angle = 0 

        for (i, scene) in enumerate(scenes):
            low = np.random.permutation(4)[:2]

            # mapping tracker_trial to the scene's trackers
            scene_data = data[data.scene == scene]
            print(scene_data)

            frames = scene_data.frame.unique()

            scene_probes = []
            for (j, frame) in enumerate(frames):
                tracker = scene_data[scene_data.frame == frame]['tracker']
                scene_probes.append([int(tracker), int(frame)])

            cond_trials.append([scene, angle, scene_probes])

        condlist.append(cond_trials)
    

    print(condlist)

    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    data = pd.read_csv('output/attention_analysis/isr_inertia_probe_map_random.csv')

    scenes = data.scene.unique().tolist()
    generate_condlist(scenes, data, os.path.join('/renders', 'condlist.json'))

    return

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import os
import json
import argparse
import pandas as pd

def generate_condlist(scenes, data, outpath):
    
    condlist = []
    
    # there are n_trackers = 3 conditions
    for tracker in range(3):
        cond_trials = []

        for probe in range(2):
            for (i, scene) in enumerate(scenes):
                probe_trial = (probe + i) % 2
                tracker_trial = (tracker + i) % 3

                # mapping tracker_trial to the scene's trackers
                scene_data = data[data.scene == scene]
                tracker_trial = scene_data.tracker.unique()[tracker_trial]
                
                query = "pr" if probe_trial == 1 else "td"
                
                # only 3 epochs (instead of 5 like before)
                for epoch in [1, 3, 5]:
                    for tf in ['T', 'F']:
                        trial = '%d_%d_t_%d_%s_%s.mp4' % (scene, tracker_trial, epoch, query, tf)
                        cond_trials.append(trial)
            
        condlist.append(cond_trials)

    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    data = pd.read_csv('output/attention_analysis/exp0_probe_map.csv')
    
    # splitting into two scene lists (every other)
    scenes = data.scene.unique()

    # kind of arbitrarily taking out the first two and the last two scenes
    # so that we have 20 scenes
    scenes = scenes[2:-2]
    
    # splitting into two batches
    scenes_1 = scenes[0::2]
    scenes_2 = scenes[1::2]
    
    generate_condlist(scenes_1, data, os.path.join('/renders', 'condlist_1.json'))
    generate_condlist(scenes_2, data, os.path.join('/renders', 'condlist_2.json'))

    return

if __name__ == '__main__':
    main()

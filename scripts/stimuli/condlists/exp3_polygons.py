#!/usr/bin/env python

import os
import json
import argparse
import pandas as pd
import numpy as np

def generate_condlist(n_scenes, outpath):
    
    condlist = []

    # just one condition for now
    for cond in range(1):
        cond_trials = []
        
        scene_probes = []

        for scene in range(1, n_scenes+1):
            cond_trials.append([scene, scene_probes])

        condlist.append(cond_trials)

    print(condlist)

    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


def main():
    parser = argparse.ArgumentParser(
        description = 'Creates condition list for experiment 3',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    with open(os.path.join('/datasets', 'exp3_polygons_v3.json')) as f:
        data = json.load(f)
        n_scenes = len(data)

    generate_condlist(n_scenes, os.path.join('/renders', 'condlist.json'))

    return

if __name__ == '__main__':
    main()

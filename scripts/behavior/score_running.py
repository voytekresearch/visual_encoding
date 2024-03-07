"""
This script loads a segmented Neo Block and scores the running behavior. This is performed for all files 
in a given folder, or list of folders. This can be used to add behavioral data to the output of 
step2_create_stimulus_segments.py if this info was not added during the initial segmentation.
"""

# SET PATH 
PATHS = [r"G:\Shared drives\visual_encoding\data\blocks\segmented\natural_movie_one_shuffled",
           r"G:\Shared drives\visual_encoding\data\blocks\lfp\natural_movie_one_more_repeats",
           r"G:\Shared drives\visual_encoding\data\blocks\lfp\natural_movie_one_shuffled"]

# imports
import os
import neo
from step2_create_stimulus_segments import score_running

# loop through folders
for path in PATHS:
    # loop through files
    files = os.listdir(path)
    for i_file, fname in enumerate(files):
        # display progress
        print(f"\nAnalyzing: {fname} ({i_file+1}/{len(files)})")

        # load block
        block = neo.io.NeoMatlabIO(filename=f"{path}/{fname}").read_block() 

        # score running
        block = score_running(block)

        # save block
        neo.io.NeoMatlabIO(filename=f"{path}/{fname}").write_block(block)
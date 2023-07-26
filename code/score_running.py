"""
This script loads a segmented Neo Block and scores the running behavior. This can be used to add
behavioral data to the outpt of step2_create_stimulus_segments.py if this info was not added during
the initial segmentation.
"""

# SET PATH
PATH_IN = r"G:\Shared drives\visual_encoding\data\blocks\segmented\natural_movie_one_more_repeats"

# imports
import os
import neo
from step2_create_stimulus_segments import score_running

# loop through files
files = os.listdir(PATH_IN)
for i_file, fname in enumerate(files):
    # display progress
    print(f"\nAnalyzing: {fname} ({i_file+1}/{len(files)})")

    # load block
    block = neo.io.NeoMatlabIO(filename=f"{PATH_IN}/{fname}").read_block() 

    # score running
    block = score_running(block)

    # save block
    neo.io.NeoMatlabIO(filename=f"{PATH_IN}/{fname}").write_block(block)
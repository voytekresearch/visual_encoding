"""
This script loads a segmented Neo Block and scores the running behavior. This is 
performed for all files in a given folder, or list of folders. This can be used 
to add behavioral data to the output of step2_create_stimulus_segments.py if 
this info was not added during the initial segmentation. Note that this
script overwrite the original blocks.
"""

# SET PATH 
BLOCK_PATHS = [r"segmented\natural_movie_one_shuffled",
                r"lfp\natural_movie_one_more_repeats",
                r"lfp\natural_movie_one_shuffled"]

# imports - standard
import os
import neo
from step2_create_stimulus_segments import score_running

# imports - custom
import sys
sys.path.append("allen_vc")
from paths import PATH_EXTERNAL


# loop through folders
for path in BLOCK_PATHS:
    # loop through files
    files = os.listdir(f"{PATH_EXTERNAL}/{path}")
    for i_file, fname in enumerate(files):
        # display progress
        print(f"\nAnalyzing: {fname} ({i_file+1}/{len(files)})")

        # load block
        block = neo.io.NeoMatlabIO(filename=f"{path}/{fname}").read_block() 

        # score running
        block = score_running(block)

        # save block
        neo.io.NeoMatlabIO(filename=f"{path}/{fname}").write_block(block)
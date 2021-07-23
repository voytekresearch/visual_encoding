"""Template script for running analysis across a group of EEG subjects, after pre-processing.
Notes:
-
-
"""

import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from funcs import *

###################################################################################################
###################################################################################################

## SETTINGS


###################################################################################################
###################################################################################################

def main():

    # Initialize any output variables to save out
    LFP_E, LFP_I, t = sim_field(4)
    LFP = LFP_E + LFP_I

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax = sns.lineplot(x = t[:5000], y = LFP_E[:5000], ax = axes[0]) # used arbitrary length of 5000 to save time
    sns.lineplot(x = t[:5000], y = LFP_I[:5000], ax = axes[0])
    ax.set_xlim(0,0.2)
    ax.legend(labels=["Excitatory","Inhibitory"])

    ax = sns.lineplot(x = t[:5000], y = LFP[:5000], color='black', ax = axes[1]) # used arbitrary length of 5000 to save time
    ax.set_xlim(0,0.2)
    ax.legend(labels=["LFP"])

    # Save any group level files
    fig.savefig('plot.png')
    


if __name__ == "__main__":
    main()
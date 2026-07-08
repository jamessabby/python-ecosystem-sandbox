# Source - https://stackoverflow.com/a
# Posted by Teddy Hartanto, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-12, License - CC BY-SA 3.0

import numpy as np
from scipy import signal # Imports signal-processing tools. We’ll use it to create a Gaussian curve.

# Kernel - A kernel is a small matrix that defines how nearby pixels influence each other.

# kernlen = kernel size (width and height)
# std = standard deviation (controls blur strength)
# So this function will create a Gaussian blur kernel.


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

import matplotlib.pyplot as plt
plt.imshow(gkern(21), interpolation='none')

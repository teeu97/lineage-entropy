import pickle
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.lines import Line2D
from matplotlib import cm
from scipy import stats

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

rainbow = cm.get_cmap('rainbow_r', 5)
rainbow_list = rainbow(range(5))

plt.figure()
cmap = matplotlib.colors.ListedColormap([rainbow_list[i] for i in range(5)][::-1])
bounds = [0, 20, 40, 60, 80, 100]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
    ticks=bounds,
    spacing='proportional',
    orientation='vertical',
    label='Percentile of Motility',
)
plt.axis('off')
plt.savefig('Colorbar_DiscreteMotility_Vertical.tiff', dpi=720, format='tiff', bbox_inches='tight')
plt.close()

plt.figure()
cmap = matplotlib.colors.ListedColormap([rainbow_list[i] for i in range(5)][::-1])
bounds = [0, 20, 40, 60, 80, 100]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
    ticks=bounds,
    spacing='proportional',
    orientation='horizontal',
    label='Percentile of Motility',
)
plt.axis('off')
plt.savefig('Colorbar_DiscreteMotility_Horizontal.tiff', dpi=720, format='tiff', bbox_inches='tight')
plt.close()
"""
This file produces a colorbar for different percentile of motility
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

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
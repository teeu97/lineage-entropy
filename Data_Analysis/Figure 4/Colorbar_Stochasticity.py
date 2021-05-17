"""
Colorbar_Stochasticity.py produces a colorbar that shows the range of stochasticity
"""
import matplotlib
import matplotlib.pyplot as plt

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                                               cmap='rainbow')

plt.figure()
plt.axis('off')
cbar = plt.colorbar(color_scalarMap, orientation='horizontal')
plt.savefig('Colorbar_Stochasticity_Horizontal.svg', dpi=720, format='svg', bbox_inches='tight')
plt.close()

plt.figure()
plt.axis('off')
cbar = plt.colorbar(color_scalarMap, orientation='vertical')
plt.savefig('Colorbar_Stochasticity_Vertical.svg', dpi=720, format='svg', bbox_inches='tight')
plt.close()



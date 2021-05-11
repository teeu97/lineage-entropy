import matplotlib
import matplotlib.pyplot as plt

color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                                               cmap='rainbow')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

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



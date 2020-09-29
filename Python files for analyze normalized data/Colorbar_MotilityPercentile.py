matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

axins = inset_axes(axs[3],
                       width="5%",
                       height="100%",
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=axs[3].transAxes,
                       borderpad=0,
                       )
    bounds = np.arange(0, 110, 10)
    cmap = matplotlib.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar = matplotlib.colorbar.ColorbarBase(axins, cmap=cmap, norm=norm)
    cbar.set_label("Motility Percentile")
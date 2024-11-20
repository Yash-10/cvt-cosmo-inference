import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.nddata import Cutout2D


import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

import h5py
def read_hdf5(filename, dtype=np.float32, dataset_name='3D_density_field'):
    hf = h5py.File(filename, 'r')
    dataset = hf.get(dataset_name)
    cosmo_params = dataset.attrs['cosmo_params']
    density_field = dataset[:]

    density_field = density_field.astype(dtype)
    cosmo_params = cosmo_params.astype(dtype)

    return density_field, cosmo_params

density, params = read_hdf5('./sim1343_LH_z0_grid256_masCIC.h5')
halo, params1 = read_hdf5('./halos_sim1343_LH_z0_grid256_masCIC.h5', dataset_name='3D_halo_distribution')

density = density[:, 10, :]
halo = halo[:, 10, :]
density = np.log10(1+density/density.mean())
halo = np.log10(1+halo/halo.mean())

assert np.all(params == params1)

vmin, vmax = 0, 2

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# fig = plt.figure(figsize=(20, 5))
# ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(1,4),
#                  axes_pad=0.15,
#                  share_all=True,
#                 #  cbar_location="right",
#                 #  cbar_mode="single",
#                 #  cbar_size="7%",
#                 #  cbar_pad=0.15,
#                  )

ax[0].imshow(density, vmin=vmin, vmax=vmax, origin='lower')
im = ax[2].imshow(halo, vmin=vmin, vmax=vmax, origin='lower')

for a in ax:
    a.set_yticks([])
    a.set_xticks([])

ax[0].set_title('DM density', fontsize=24)
ax[2].set_title('Halo', fontsize=24)

# axins = zoomed_inset_axes(ax[0], 3, loc='lower left', bbox_to_anchor=(705, 145), borderpad=3) # zoom = 6
cut = Cutout2D(density, (80, 150), size=50)
# axins.imshow(cut.data, interpolation="nearest", origin="lower")
# x1, x2, y1, y2 = 0, 49, 0, 49
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticks([]); axins.set_yticks([])
cut.plot_on_original(ax[0], color='white', linewidth=1.)
ax[1].imshow(cut.data, origin='lower')

# axins = zoomed_inset_axes(ax[1], 3, loc='lower left', bbox_to_anchor=(705, 0), borderpad=3) # zoom = 6
cut = Cutout2D(halo, (80, 150), size=50)
# axins.imshow(cut.data, interpolation="nearest", origin="lower")
# x1, x2, y1, y2 = 0, 49, 0, 49
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticks([]); axins.set_yticks([])
cut.plot_on_original(ax[2], color='white', linewidth=1.)
ax[3].imshow(cut.data, origin='lower')

# divider = make_axes_locatable(ax[3])
# cax1 = divider.append_axes("right", size="5%", pad=0.05)
# cax1 = make_square_axes_with_colorbar(ax[3], size=0.15, pad=0.05)

for i, a in enumerate(ax):
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad='5%')
    if i <= 2:
        cax.set_axis_off()

cbar = plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad = 30
cbar.ax.set_ylabel(r'$log(1 + \frac{\rho}{\bar{\rho}})$', rotation=270)

fig.tight_layout()

plt.savefig('output.pdf', format='pdf')

plt.show()

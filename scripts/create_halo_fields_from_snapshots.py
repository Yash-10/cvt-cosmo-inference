# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Reads QUIJOTE snapshots and interpolates the particle positions
#              on a predefined grid. [Halo]

# ***************************************************************************
import h5py
import hdf5plugin
import gzip
import os
import numpy as np
import readfof
import MAS_library as MASL

import matplotlib.pyplot as plt
import subprocess

# CONSTANTS
NUM_SIMS = 2000  # Can be any integer <= 2000.
BASE_PATH = './halos'
OUTPUT_DIR = './my_outputs_halo_128/'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

grid     = 128    #the density field will have grid^3 voxels
MAS      = 'CIC'  #Mass-assignment scheme:'NGP', 'CIC', 'TSC', 'PCS'
verbose  = True   #whether to print information about the progress
ptype    = [1]    #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

def calculate_HMF(mass_h, Np_h, min_mass=2e13, max_mass=1e15):  # min_mass and max_mass in Msun/h.
    # min_mass = 2e13 #minimum mass in Msun/h
    # max_mass = 1e15 #maximum mass in Msun/h
    bins     = 30   #number of bins in the HMF

    # Correct the masses of the FoF halos
    mass_h = mass_h*(1.0-Np_h**(-0.6))

    bins_mass = np.logspace(np.log10(min_mass), np.log10(max_mass), bins+1)
    mass_mean = 10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
    dM        = bins_mass[1:] - bins_mass[:-1]

    # compute the halo mass function (number of halos per unit volume per unit mass)
    HMF = np.histogram(mass_h, bins=bins_mass)[0]/(dM*BoxSize**3)

    return mass_mean, HMF

import sys
save_stdout = sys.stdout
sys.stdout = open('trash', 'w')

bad_flags = []  # For storing simulation numbers having empty voxels in the density field.

masses = []
HMFs = []
lh_params = np.loadtxt(os.path.join(BASE_PATH, "latin_hypercube_params.txt"))

for i in range(NUM_SIMS):
    snapdir = f'./halos/{i}' #folder hosting the catalogue
#     snapnum = 4                                      #number of the catalog (4-->z=0, 3-->z=0.5, 2-->z=1, 1-->z=2, 0-->z=3)

    cosmo_params = lh_params[i]  # For halos, we do not load the parameters from Cosmo_params.dat (as in density field construction) because there is no such file for Halos. So we load all parameters at once from the text file.

#     header   = readgadget.header(snapshot)
#     BoxSize  = header.boxsize/1e3  #Mpc/h
#     redshift = header.redshift     #redshift of the snapshot
#     Masses   = header.massarr*1e10 #Masses of the particles in Msun/h

    # We are hardcoding these values since it's not possible to access both the Snapshots and the Halo catalogs
    # from either the San Diego or New York cluster for the latin hypercube data.
    # TODO: I have asked Quijote simulations creators if this is the right thing to do. Until then, I don't see any problems.
    redshift = 2.220446049250313e-16
    BoxSize = 1000.0

    # read the halo catalogue
    FoF = readfof.FoF_catalog(snapdir, snapnum=4, long_ids=False,  # 4 means z=0.
                              swap=False, SFR=False, read_IDs=False)

    # get the properties of the halos
    pos_h  = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    vel_h  = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s
    mass_h = FoF.GroupMass*1e10          #Halo masses in Msun/h
    Np_h   = FoF.GroupLen                #Number of CDM particles in the halo. Even in simulations with massive neutrinos, this will be just the number of CDM particles

    delta = np.zeros((grid,grid,grid), dtype=np.float32)

    # Calculate HMF.
    mass_mean, HMF = calculate_HMF(mass_h, Np_h, min_mass=10**12.5, max_mass=10**16.5)
    masses.append(mass_mean)
    HMFs.append(HMF)

    # Construct 3D halo distribution
    MASL.MA(pos_h, delta, BoxSize, MAS, verbose=verbose)

    # We want the effective no. of particles in each voxel, so skip the below code.
#     delta *= Masses[1]

    if np.any(delta == 0.0):
        print('Halo distribution on the grid contains at least one empty voxel!')
        bad_flags.append(1)
    else:
        bad_flags.append(0)

    filename = os.path.join(OUTPUT_DIR, f'halos_sim{i}_LH_z0_grid{grid}_mas{MAS}.h5')
    h5f = h5py.File(filename, 'w')
    dataset = h5f.create_dataset('3D_halo_distribution', data=delta, compression='gzip')
    dataset.attrs['cosmo_params'] = cosmo_params  # Order of storing parameters is same as Cosmo_params.dat
    h5f.close()

sys.stdout = save_stdout

subprocess.run(['tar', '-czf', f'halo_maps_3D_LH_z0_grid{grid}_mas{MAS}.tar.gz', 'my_outputs'])
subprocess.run(['ls', '-lh', f'halo_maps_3D_LH_z0_grid{grid}_mas{MAS}.tar.gz'])

assert not np.any(bad_flags == 1)

# Show an example halo map and HMF.
plt.figure(figsize=(8, 8)); plt.imshow(delta[:,:,0]); plt.colorbar(); plt.savefig(f'halo_map_example_grid{grid}_MAS{MAS}.png', bbox_inches='tight', dpi=200)
plt.clf()

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm halo}~[h^{-1}M_\odot]$')
plt.ylabel(r'$HMF~[h^4M_\odot^{-1}{\rm Mpc}^{-3}]$')
plt.plot(np.mean(masses, axis=0), np.mean(HMFs, axis=0))
plt.show()

# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Reads QUIJOTE snapshots and interpolates the particle positions
#              on a predefined grid.

# ***************************************************************************
import h5py
import hdf5plugin
import gzip
import os
import numpy as np
import readgadget
import MAS_library as MASL

import matplotlib.pyplot as plt
import subprocess

# CONSTANTS
istart = 426
NUM_SIMS = 434  # Can be any integer <= 2000.
# The below path is for the 2000 standard latin hypercube simulations.
# Since it's 'standard', 512^3 particles are present, and since it's QUIJOTE, L_box = 1 Gpc h^-1.
# NOTE: Change the below path (BASE_PATH) depending on your local path where the snapshots are stored.
# Inside this folder, there must be folders named 0, 1, 2, ..., inside which there must be a folder named "snapdir_004", inside which there will be multiple HDF5 files.
BASE_PATH = './snapshots' #/latin_hypercube'
OUTPUT_DIR = './my_outputs/'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

grid     = 256    #the density field will have grid^3 voxels
MAS      = 'CIC'  #Mass-assignment scheme:'NGP', 'CIC', 'TSC', 'PCS'
verbose  = True   #whether to print information about the progress
ptype    = [1]    #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

bad_flags = []  # For storing simulation numbers having empty voxels in the density field.

for i in range(istart,NUM_SIMS):
    print(f"reading {i}")
    snapshot = os.path.join(BASE_PATH, f'{i}', "snap_004") #, 'snapdir_004', 'snap_004')
    #cosmo_params = np.loadtxt(
    #    os.path.join(BASE_PATH, f'{i}', 'Cosmo_params.dat')
    #)
    cosmo_params = np.loadtxt(
        os.path.join(BASE_PATH, "latin_hypercube_params.txt")
    )[i]

    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    redshift = header.redshift     #redshift of the snapshot
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h

    # Read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h

    delta = np.zeros((grid,grid,grid), dtype=np.float32)

    # Construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

    # We want the effective no. of particles in each voxel, so skip the below code.
#     delta *= Masses[1]

    if np.any(delta == 0.0):
        print('Density field contains at least one empty voxel!')
        bad_flags.append(1)
    else:
        bad_flags.append(0)

    filename = os.path.join(OUTPUT_DIR, f'sim{i}_LH_z0_grid{grid}_mas{MAS}.h5')
    h5f = h5py.File(filename, 'w')
    dataset = h5f.create_dataset('3D_density_field', data=delta, compression='gzip')
    dataset.attrs['cosmo_params'] = cosmo_params  # Order of storing parameters is same as Cosmo_params.dat
    h5f.close()

subprocess.run(['tar', '-czf', f'density_fields_3D_LH_z0_grid{grid}_mas{MAS}.tar.gz', 'my_outputs'])
subprocess.run(['ls', '-lh', f'density_fields_3D_LH_z0_grid{grid}_mas{MAS}.tar.gz'])

assert not np.any(bad_flags == 1)

# Show an example density field
plt.figure(figsize=(8, 8)); plt.imshow(np.log10(delta[:,:,0])); plt.colorbar(); plt.savefig(f'density_field_example_grid{grid}_MAS{MAS}.png', bbox_inches='tight', dpi=200)

import os
import numpy as np
import argparse

from utils import print_options, preprocess_a_map, read_hdf5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sets preprocessing options')
    parser.add_argument('--num_sims', type=int, default=1000, help='No. of simulations to use')
    parser.add_argument('--train_frac', type=int, default=0.8, help='The fraction of simulations to use for training.')
    parser.add_argument('--test_frac', type=int, default=0.1, help='The fraction of simulations to use for testing. If train_frac + test_frac != 1.0, then the remaining are stored as validation set.')
    parser.add_argument('--path', type=str, default=None,
        help='Path to the folder containing the 3D density fields for n simulations. This folder must contain n folders with names [0 to n-1], and each folder must contain a HDF5 file storing the density field.'
    )
    parser.add_argument('--output_folder_name', type=str, default='train', help='Name of training folder where processed outputs are stored.')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    den_arr = []
    cosmo_arr = []
    for i in range(opt.num_sims):
        density, cosmo_params = read_hdf5(os.path.join(opt.path, f'{i}', 'sim{i}_LH_z0_grid64_masCIC.h5'))
        den_arr.append(density)
        cosmo_arr.append(cosmo_params)

    mean, std = np.mean(density), np.std(density)
    del den_arr, cosmo_arr

    for i in range(opt.num_sims):
        density, cosmo_params = read_hdf5(os.path.join(opt.path, f'{i}', 'sim{i}_LH_z0_grid64_masCIC.h5'))
        density = reprocess_a_map(density, mean=mean, std=std)
        cosmo_params = normalize_cosmo_param(cosmo_params, min_vals=min_vals, max_vals=max_vals)

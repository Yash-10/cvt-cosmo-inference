import os
import h5py
import numpy as np
import argparse
import shutil

from utils import print_options, preprocess_a_map, read_hdf5, normalize_cosmo_param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sets preprocessing options')
    parser.add_argument('--num_sims', type=int, default=1000, help='No. of simulations to use')
    parser.add_argument('--train_frac', type=float, default=0.8, help='The fraction of simulations to use for training.')
    parser.add_argument('--test_frac', type=float, default=0.1, help='The fraction of simulations to use for testing. If train_frac + test_frac != 1.0, then the remaining are stored as validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use for splitting data into train, test, and val sets.')
    parser.add_argument('--path', type=str, default=None,
        help='Path to the folder containing the 3D density fields for n simulations. This folder must contain n folders with names [0 to n-1], and each folder must contain a HDF5 file storing the density field.'
    )
    # parser.add_argument('--output_folder_name', type=str, default='train', help='Name of training folder where processed outputs are stored.')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    den_arr = []
    cosmo_arr = []
    for i in range(opt.num_sims):
        density, cosmo_params = read_hdf5(os.path.join(opt.path, f'sim{i}_LH_z0_grid64_masCIC.h5'))
        den_arr.append(np.log10(density))
        cosmo_arr.append(cosmo_params)

    mean, std = np.mean(density), np.std(density)
    del den_arr

    cosmo_arr = np.array(cosmo_arr)
    min_vals = np.array([cosmo_arr[:, i].min() for i in range(5)])  # 5 parameters.
    max_vals = np.array([cosmo_arr[:, i].max() for i in range(5)])  # 5 parameters.

    # Decide splits for train, test, and val
    np.random.seed(opt.seed)
    sim_numbers = np.arange(opt.num_sims)
    np.random.shuffle(sim_numbers)

    end = round(opt.train_frac*opt.num_sims)
    train_sim_numbers = sim_numbers[:end]
    end2 = round(opt.test_frac*opt.num_sims)
    test_sim_numbers = sim_numbers[end:end+end2]

    val = opt.train_frac + opt.test_frac != 1.0
    if val:  # It means validation set must also be saved.
        val_sim_numbers = sim_numbers[end+end2:]

    if os.path.exists('train'):
        shutil.rmtree('train')
    if os.path.exists('test'):
        shutil.rmtree('test')
    if os.path.exists('val'):
        shutil.rmtree('val')

    os.mkdir('test')
    os.mkdir('train')
    os.mkdir('val')

    for i in range(opt.num_sims):
        density, cosmo_params = read_hdf5(os.path.join(opt.path, f'sim{i}_LH_z0_grid64_masCIC.h5'))
        density = preprocess_a_map(density, mean=mean, std=std)
        cosmo_params = normalize_cosmo_param(cosmo_params, min_vals=min_vals, max_vals=max_vals)

        if i in train_sim_numbers:
            filename = os.path.join('train', f'processed_sim{i}_LH_z0_grid64_masCIC.h5')
        elif i in test_sim_numbers:
            filename = os.path.join('test', f'processed_sim{i}_LH_z0_grid64_masCIC.h5')
        elif val:
            if i in val_sim_numbers:
                filename = os.path.join('val', f'processed_sim{i}_LH_z0_grid64_masCIC.h5')

        h5f = h5py.File(filename, 'w')
        dataset = h5f.create_dataset('3D_density_field', data=density, compression='gzip')
        dataset.attrs['cosmo_params'] = cosmo_params  # Order of storing parameters is same as Cosmo_params.dat
        h5f.close()

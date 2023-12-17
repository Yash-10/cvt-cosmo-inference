import os
import h5py
import pandas as pd
import numpy as np
import argparse
import random
import shutil
import gzip

from utils import print_options, preprocess_a_map, read_hdf5, normalize_cosmo_param, approx_lte, approx_gte, calculate_overdensity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sets preprocessing options')
    parser.add_argument('--num_sims', type=int, default=1000, help='No. of simulations to use')
    parser.add_argument('--grid_size', type=int, default=64, help='Grid size for the density fields that need to be procesed and saved.')
    parser.add_argument('--train_frac', type=float, default=0.8, help='The fraction of simulations to use for training.')
    parser.add_argument('--test_frac', type=float, default=0.1, help='The fraction of simulations to use for testing. If train_frac + test_frac != 1.0, then the remaining are stored as validation set.')
    parser.add_argument('--num_maps_per_projection_direction', type=int, help='No. of maps to select per projection direction. Total no. of maps is three times this number.')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use for splitting data into train, test, and val sets.')
    parser.add_argument('--path', type=str, default=None,
        help='Path to the folder containing the 3D density fields for n simulations. This folder must contain n folders with names [0 to n-1], and each folder must contain a HDF5 file storing the density field.'
    )
    parser.add_argument('--precomputed_mean', type=float, default=np.nan, help='Precomputed mean to use for preprocessing. This is helpful for transfer learning/fine-tuning.')
    parser.add_argument('--precomputed_stddev', type=float, default=np.nan, help='Precomputed standard deviation to use for preprocessing. This is helpful for transfer learning/fine-tuning.')
    parser.add_argument('--precomputed_min_vals', action='store', type=float, nargs=5)
    parser.add_argument('--precomputed_max_vals', action='store', type=float, nargs=5)
    parser.add_argument('--prefix', type=str, default='', help='Prefix in the filenames.')
    parser.add_argument('--dataset_name', type=str, default='3D_density_field', help='Name of the HDF5 dataset.')
    parser.add_argument('--log_1_plus', action='store_true', help='If specified, take log10 of 1 + pixel_value instead of log10 of pixel value.')
    parser.add_argument('--bias', type='float', default=np.nan, help='Value of bias parameter. This is used when preparing data for transfer learning on halo catalog data after pretraining on DM density.')

    # parser.add_argument('-precomputed_min_vals', '--list', nargs='+', help='Precomputed minimum values of the parameters. This is helpful for transfer learning/fine-tuning.')
    # parser.add_argument('-precomputed_max_vals', '--list', nargs='+', help='Precomputed maximum values of the parameters. This is helpful for transfer learning/fine-tuning.')
    # parser.add_argument('--output_folder_name', type=str, default='train', help='Name of training folder where processed outputs are stored.')

    opt = parser.parse_args()

    dtype = np.float32

    # Print options
    print_options(opt)

    if opt.num_maps_per_projection_direction > opt.grid_size:
        raise ValueError("No. of 2D maps to extract per dimension exceeds the grid size!")

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

    assert set(train_sim_numbers).isdisjoint(test_sim_numbers)
    if val:
        assert set(test_sim_numbers).isdisjoint(val_sim_numbers)
        assert set(train_sim_numbers).isdisjoint(val_sim_numbers)

    if os.path.exists('train'):
        shutil.rmtree('train')
    if os.path.exists('test'):
        shutil.rmtree('test')
    if os.path.exists('val'):
        shutil.rmtree('val')

    os.mkdir('test')
    os.mkdir('train')
    os.mkdir('val')

    # Calculate statistics across training set for normalization.
    den_arr = []
    cosmo_arr = []
    for i in range(opt.num_sims):
        if i in train_sim_numbers:  # We want to calculate statistics only using the training set.
            suffix = f'sim{i}_LH_z0_grid{opt.grid_size}_masCIC.h5' if opt.prefix == '' else f'{opt.prefix}_sim{i}_LH_z0_grid{opt.grid_size}_masCIC.h5'
            density, cosmo_params = read_hdf5(os.path.join(opt.path, suffix), dtype=dtype, dataset_name=opt.dataset_name)
            # Calculate the overdensity field: 1 + delta.
            density = calculate_overdensity(density)
            if not np.isnan(opt.bias):  # This means we are preparing data for transfer learning.
                density = (density + (opt.bias - 1)) / opt.bias
            # Assert it's a cube
            assert density.shape[0] == density.shape[1]
            assert density.shape[0] == density.shape[2]
            # Assert the grid size passed via commandline and the density field shape match.
            assert density.shape[0] == opt.grid_size
            den_arr.append(np.log10(1+density) if opt.log_1_plus else np.log10(density))
            cosmo_arr.append(cosmo_params)

    if np.isnan(opt.precomputed_mean) and np.isnan(opt.precomputed_stddev):
        mean, std = np.mean(den_arr), np.std(den_arr)
    elif ~np.isnan(opt.precomputed_mean) and ~np.isnan(opt.precomputed_stddev):
        mean, std = opt.precomputed_mean, opt.precomputed_stddev
    else:
        raise ValueError("One of mean/stddev is specified but the other is not specified. Either specify both or don't specify both.")

    del den_arr

    if opt.precomputed_min_vals is None and opt.precomputed_max_vals is None:
        cosmo_arr = np.array(cosmo_arr, dtype=dtype)
        min_vals = np.array([cosmo_arr[:, i].min() for i in range(5)])  # 5 parameters.
        max_vals = np.array([cosmo_arr[:, i].max() for i in range(5)])  # 5 parameters.
    elif opt.precomputed_min_vals is not None and opt.precomputed_max_vals is not None:
        min_vals = np.array(opt.precomputed_min_vals)
        max_vals = np.array(opt.precomputed_max_vals)
    else:
        raise ValueError("One of min_vals/max_vals is specified but the other is not specified. Either specify both or don't specify both.")

    train_param_file = os.path.join('train', 'train_normalized_cosmo_params_train.txt')
    train_orig_param_file = os.path.join('train', 'train_orig_cosmo_params_train.txt')
    test_param_file = os.path.join('test', 'test_normalized_cosmo_params_train.txt')
    test_orig_param_file = os.path.join('test', 'test_orig_cosmo_params_train.txt')
    if val:
        val_param_file = os.path.join('val', 'val_normalized_cosmo_params_train.txt')
        val_orig_param_file = os.path.join('val', 'val_orig_cosmo_params_train.txt')

    if os.path.exists(train_param_file):
        os.remove(train_param_file)
    if os.path.exists(train_orig_param_file):
        os.remove(train_orig_param_file)
    if os.path.exists(test_param_file):
        os.remove(test_param_file)
    if os.path.exists(test_orig_param_file):
        os.remove(test_orig_param_file)
    if os.path.exists(val_param_file):
        os.remove(val_param_file)
    if os.path.exists(val_orig_param_file):
        os.remove(val_orig_param_file)

    # tr = open(train_param_file, 'a')
    # tro = open(train_orig_param_file, 'a')
    # te = open(train_param_file, 'a')
    # teo = open(test_orig_param_file, 'a')
    # va = open(train_param_file, 'a')
    # vao = open(val_orig_param_file, 'a')

    train_param_data = []
    test_param_data = []
    orig_train_param_data = []
    orig_test_param_data = []
    if val:
        val_param_data = []
        orig_val_param_data = []

    random.seed(opt.seed)
    slice_indices_list = [random.sample(range(density.shape[0]), k=opt.num_maps_per_projection_direction) for _ in range(opt.num_sims)]

    for i in range(opt.num_sims):
        suffix = f'sim{i}_LH_z0_grid{opt.grid_size}_masCIC.h5' if opt.prefix == '' else f'{opt.prefix}_sim{i}_LH_z0_grid{opt.grid_size}_masCIC.h5'
        density, cosmo_params = read_hdf5(os.path.join(opt.path, suffix), dtype=dtype, dataset_name=opt.dataset_name)
        # Calculate overdensity.
        density = calculate_overdensity(density)
        if not np.isnan(opt.bias):  # This means we are preparing data for transfer learning.
            density = (density + (opt.bias - 1)) / opt.bias
        density = preprocess_a_map(density, mean=mean, std=std, log_1_plus=opt.log_1_plus)
        normalized_cosmo_params = normalize_cosmo_param(cosmo_params, min_vals=min_vals, max_vals=max_vals)

        if i in train_sim_numbers:
            # This condition necessarily needs to hold for training data since the normalization took place using training parameter values.
            # But it may not hold for val/test sets. So we don't check for this condition in those cases.
            assert approx_gte(normalized_cosmo_params.min(), 0) and approx_lte(normalized_cosmo_params.max(), 1)  # Due to the min-max normalization.

        # Now extract 2D maps from the 3D field
        # Assert it's a cube.
        assert density.shape[0] == density.shape[1]
        assert density.shape[0] == density.shape[2]

        normalized_cosmo_params = normalized_cosmo_params.astype(object)
        cosmo_params = cosmo_params.astype(object)

        slice_indices = slice_indices_list[i]

        for j in slice_indices:
            if i in train_sim_numbers:
                filename1 = os.path.join('train', f'processed_sim{i}_X{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                filename2 = os.path.join('train', f'processed_sim{i}_Y{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                filename3 = os.path.join('train', f'processed_sim{i}_Z{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                for fn in [filename1, filename2, filename3]:
                    train_param_data.append(np.insert(normalized_cosmo_params, 0, fn))
                    orig_train_param_data.append(np.insert(cosmo_params, 0, fn))
                # for fn in [filename1, filename2, filename3]:
                #     tr.write([fn, normalized_cosmo_params])
                #     tro.write([fn, cosmo_params])
            elif i in test_sim_numbers:
                filename1 = os.path.join('test', f'processed_sim{i}_X{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                filename2 = os.path.join('test', f'processed_sim{i}_Y{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                filename3 = os.path.join('test', f'processed_sim{i}_Z{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                for fn in [filename1, filename2, filename3]:
                    test_param_data.append(np.insert(normalized_cosmo_params, 0, fn))
                    orig_test_param_data.append(np.insert(cosmo_params, 0, fn))
                # for fn in [filename1, filename2, filename3]:
                #     te.write([fn, normalized_cosmo_params])
                #     teo.write([fn, cosmo_params])
            elif val:
                if i in val_sim_numbers:
                    filename1 = os.path.join('val', f'processed_sim{i}_X{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                    filename2 = os.path.join('val', f'processed_sim{i}_Y{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                    filename3 = os.path.join('val', f'processed_sim{i}_Z{j}_LH_z0_grid{opt.grid_size}_masCIC.npy.gz')
                    for fn in [filename1, filename2, filename3]:
                        val_param_data.append(np.insert(normalized_cosmo_params, 0, fn))
                        orig_val_param_data.append(np.insert(cosmo_params, 0, fn))
                    # for fn in [filename1, filename2, filename3]:
                    #     va.write([fn, normalized_cosmo_params])
                    #     vao.write([fn, cosmo_params])

            f = gzip.GzipFile(filename1, 'w')
            np.save(file=f, arr=density[j, :, :])
            f.close()
            del f

            f = gzip.GzipFile(filename2, 'w')
            np.save(file=f, arr=density[:, j, :])
            f.close()
            del f

            f = gzip.GzipFile(filename3, 'w')
            np.save(file=f, arr=density[:, :, j])
            f.close()
            del f
    
    train_param_data = pd.DataFrame(train_param_data)
    orig_train_param_data = pd.DataFrame(orig_train_param_data)
    test_param_data = pd.DataFrame(test_param_data)
    orig_test_param_data = pd.DataFrame(orig_test_param_data)
    if val:
        val_param_data = pd.DataFrame(val_param_data)
        orig_val_param_data = pd.DataFrame(orig_val_param_data)

    train_param_data.to_csv(os.path.join('train', 'train_normalized_params.csv'))
    orig_train_param_data.to_csv(os.path.join('train', 'train_original_params.csv'))
    test_param_data.to_csv(os.path.join('test', 'test_normalized_params.csv'))
    orig_test_param_data.to_csv(os.path.join('test', 'test_original_params.csv'))
    if val:
        val_param_data.to_csv(os.path.join('val', 'val_normalized_params.csv'))
        orig_val_param_data.to_csv(os.path.join('val', 'val_original_params.csv'))

    print(f'Mean of log10(den) across the training set: {mean}')
    print(f'Std. dev of log10(den) across the training set: {std}')
    print(f'Min values of parameters across the training set: {min_vals}')
    print(f'Max values of parameters across the training set: {max_vals}')

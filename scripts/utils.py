import os
import numpy as np
import glob
import h5py

def print_options(opt):
    print('\n')
    print("------------ Options ------------")
    for arg in vars(opt):
        print(f'{arg}:\t\t{getattr(opt, arg)}')
    print("------------ End ------------")
    print('\n')

def preprocess_a_map(map, mean=None, std=None):
    # Take logarithm.
    map = np.log10(map)
    if mean is None or std is None:
        raise ValueError(
            "Both mean and std must not be None. Please calculate the mean and std across the training set and pass these values to this function."
        )

    map = (map - mean) / std
    return map

def normalize_cosmo_param(param, min_vals, max_vals):
    # param is a 1d array with six entries. Similarly for min_vals and max_vals.
    return (param - min_vals) / (max_vals - min_vals)

def read_hdf5(filename, dtype=np.float32):
    hf = h5py.File(filename, 'r')
    dataset = hf.get('3D_density_field')
    cosmo_params = dataset.attrs['cosmo_params']
    density_field = dataset[:]

    density_field = density_field.astype(dtype)
    cosmo_params = cosmo_params.astype(dtype)

    return density_field, cosmo_params

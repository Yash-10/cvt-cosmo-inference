import os
import numpy as np
import glob
import pandas as pd
import h5py
from sklearn.metrics import r2_score, mean_squared_error

import random
import torch
import torchvision.transforms.functional as TF

import yaml

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config['params'] #[0,1,2,3,4]           
g = params
h = [5+i for i in g]

def extract_model_params(filename, key):
    with open(filename, 'r') as f:
        lines = f.readlines()
    key_line_index = None
    for i, line in enumerate(lines):
        if key in line:
            key_line_index = i
            break

    if key_line_index is None:
        raise ValueError(f"Key {key} not found in file {filename}")
    
    extracted_lines = [lines[key_line_index]]
    for line in lines[key_line_index+1:]:
        if line.startswith(" ") or line.startswith("\t"):
            extracted_lines.append(line)
        else:   
            break

    extracted_yaml = ''.join(extracted_lines)  
    extracted_data = yaml.safe_load(extracted_yaml)
    return extracted_data [key]

# See https://pytorch.org/vision/0.15/transforms.html
class MyRotationTransform:
    """Rotate by 90/180/270 degrees."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        x = torch.from_numpy(x).unsqueeze(0)
        return TF.rotate(x, angle).squeeze()

class MyFilterSmallValues:
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, x):
        x = torch.where(x < self.threshold, torch.tensor(self.threshold, dtype=x.dtype), x)
        return x

from torch.utils.data.dataset import Dataset
import gzip
class CustomImageDataset(Dataset):
    def __init__(self, img_folder_path, normalized_cosmo_params_path, transform=None, nmax=None):
            self.normalized_cosmo_params_path = normalized_cosmo_params_path
            self.normalized_cosmo_params = pd.read_csv(self.normalized_cosmo_params_path)
            self.img_folder_path = img_folder_path
            self.transform = transform

            if nmax is not None:
                self.normalized_cosmo_params = self.normalized_cosmo_params.iloc[:nmax]

    def __len__(self):
            return len(self.normalized_cosmo_params)

    def __getitem__(self, idx):
            img_path = self.normalized_cosmo_params.iloc[idx, 1]
            f = gzip.GzipFile(img_path, 'r')
            image = np.load(f)
            label = np.array(self.normalized_cosmo_params.iloc[idx, -5:], dtype=np.float32)
            label = label[g]
            if self.transform:
                image = self.transform(image)
                image = image.to(torch.float32)

            image = np.expand_dims(image, 0)
            return image, label, img_path

# Below two from https://stackoverflow.com/a/55945030
def approx_lte(x, y):
    return x <= y or np.isclose(x, y, atol=1e-5)

def approx_gte(x, y):
    return x >= y or np.isclose(x, y, atol=1e-5)

def print_options(opt):
    print('\n')
    print("------------ Options ------------")
    for arg in vars(opt):
        print(f'{arg}:\t\t{getattr(opt, arg)}')
    print("------------ End ------------")
    print('\n')

def preprocess_a_map(map, mean=None, std=None, log_1_plus=False):
    map = np.log10(1 + map) if log_1_plus else np.log10(map)
    if mean is None or std is None:
        raise ValueError(
            "Both mean and std must not be None. Please calculate the mean and std across the training set and pass these values to this function."
        )

    # Standardize map.
    map = (map - mean) / std
    return map

def unprocess_a_map(map, mean, std, original_density_mean, log_1_plus=False, bias=np.nan):
    """_summary_

    Args:
        map (_type_): 2D map.
        mean (_type_): Mean of {overdensity+[log10(map) or log10(1+map)]} values across the training set.
        std (_type_): Std. dev of {overdensity+[log10(map) or log10(1+map)]} values across the training set.
        original_density_mean (_type_): Mean of densities in the 2D map of the original map.
        log_1_plus (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    unstandardized_map = map * std + mean
    unstandardized_map = 10 ** unstandardized_map - 1 if log_1_plus else 10 ** unstandardized_map
    if not np.isnan(bias):
        unstandardized_map = unstandardized_map * bias - (bias - 1)
    original_density = unstandardized_map * original_density_mean
    return original_density

def normalize_cosmo_param(param, min_vals, max_vals):
    # param is a 1d array with six entries. Similarly for min_vals and max_vals.
    return (param - min_vals) / (max_vals - min_vals)

# def unprocess_a_map(map, mean, std, log_1_plus=False, bias=np.nan):
#     if not np.isnan(bias):
#         map = map * bias - (bias - 1)
#     x = (map * std + mean)
#     return 10**x-1 if log_1_plus else 10**x

def read_hdf5(filename, dtype=np.float32, dataset_name='3D_density_field'):
    hf = h5py.File(filename, 'r')
    dataset = hf.get(dataset_name)
    cosmo_params = dataset.attrs['cosmo_params']
    density_field = dataset[:]

    density_field = density_field.astype(dtype)
    cosmo_params = cosmo_params.astype(dtype)

    return density_field, cosmo_params


def calculate_overdensity(arr):
    return arr / np.mean(arr)

# CKA code
# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Efficient implementation of CKA based on minibatch statistics"""

from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


class MinibatchCKA(tf.keras.metrics.Metric):

    def __init__(self,
                num_layers,
                num_layers2=None,
                across_models=False,
                dtype=tf.float32):
        super(MinibatchCKA, self).__init__()
        if num_layers2 is None:
            num_layers2 = num_layers
        self.hsic_accumulator = self.add_weight(
            'hsic_accumulator',
            shape=(num_layers, num_layers2),
            initializer=tf.keras.initializers.zeros,
            dtype=dtype)
        self.across_models = across_models
        if across_models:
            self.hsic_accumulator_model1 = self.add_weight(
                'hsic_accumulator_model1',
                shape=(num_layers,),
                initializer=tf.keras.initializers.zeros,
                dtype=dtype)
            self.hsic_accumulator_model2 = self.add_weight(
                'hsic_accumulator_model2',
                shape=(num_layers2,),
                initializer=tf.keras.initializers.zeros,
                dtype=dtype)

    def _generate_gram_matrix(self, x):
        """Generate Gram matrix and preprocess to compute unbiased HSIC.

        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.

        Args:
            x: A [num_examples, num_features] matrix.

        Returns:
            A [num_examples ** 2] vector.
        """
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(x, x, transpose_b=True)
        n = tf.shape(gram)[0]
        gram = tf.linalg.set_diag(gram, tf.zeros((n,), gram.dtype))
        gram = tf.cast(gram, self.hsic_accumulator.dtype)
        means = tf.reduce_sum(gram, 0) / tf.cast(n - 2, self.hsic_accumulator.dtype)
        means -= tf.reduce_sum(means) / tf.cast(2 * (n - 1),
                                                self.hsic_accumulator.dtype)
        gram -= means[:, None]
        gram -= means[None, :]
        gram = tf.linalg.set_diag(gram, tf.zeros((n,), self.hsic_accumulator.dtype))
        gram = tf.reshape(gram, (-1,))
        return gram

    def update_state(self, activations):
        """Accumulate minibatch HSIC values.

        Args:
            activations: A list of activations for all layers.
        """
        # tf.assert_equal(
        #     tf.shape(self.hsic_accumulator)[0], len(activations),
        #     'Number of activation vectors does not match num_layers.')
        layer_grams = [self._generate_gram_matrix(x) for x in activations]
        layer_grams = tf.stack(layer_grams, 0)
        self.hsic_accumulator.assign_add(
            tf.matmul(layer_grams, layer_grams, transpose_b=True))

    def update_state_across_models(self, activations1, activations2):
        """Accumulate minibatch HSIC values from different models.

        Args:
            activations1: A list of activations for all layers in model 1.
            activations2: A list of activations for all layers in model 2.
        """
        tf.assert_equal(
            tf.shape(self.hsic_accumulator)[0], len(activations1),
            'Number of activation vectors does not match num_layers.')
        tf.assert_equal(
            tf.shape(self.hsic_accumulator)[1], len(activations2),
            'Number of activation vectors does not match num_layers.')
        layer_grams1 = [self._generate_gram_matrix(x) for x in activations1]
        layer_grams1 = tf.stack(layer_grams1, 0)  #(n_layers, n_examples ** 2)
        layer_grams2 = [self._generate_gram_matrix(x) for x in activations2]
        layer_grams2 = tf.stack(layer_grams2, 0)
        self.hsic_accumulator.assign_add(
            tf.matmul(layer_grams1, layer_grams2, transpose_b=True))
        self.hsic_accumulator_model1.assign_add(
            tf.einsum('ij,ij->i', layer_grams1, layer_grams1))
        self.hsic_accumulator_model2.assign_add(
            tf.einsum('ij,ij->i', layer_grams2, layer_grams2))

    def result(self):
        mean_hsic = tf.convert_to_tensor(
            self.hsic_accumulator)  #(num_layers, num_layers2)
        if self.across_models:
            normalization1 = tf.sqrt(
                tf.convert_to_tensor(self.hsic_accumulator_model1))  #(num_layers,)
            normalization2 = tf.sqrt(
                tf.convert_to_tensor(self.hsic_accumulator_model2))  #(num_layers2,)
            mean_hsic /= normalization1[:, None]
            mean_hsic /= normalization2[None, :]
        else:
            normalization = tf.sqrt(tf.linalg.diag_part(mean_hsic))
            mean_hsic /= normalization[:, None]
            mean_hsic /= normalization[None, :]
        return mean_hsic


def test_CKA(n_layers,
             n_layers2,
             activations1,
             activations2,
             cka1=None,
             cka2=None):
    """Test for checking that update_state_across_models() works as intended"""
    if cka1 is None:
        cka1 = MinibatchCKA(n_layers, n_layers2, across_models=True)
    if cka2 is None:
        cka2 = MinibatchCKA(n_layers + n_layers2)

    cka1.update_state_across_models(activations1, activations2)
    cka1_result = cka1.result().numpy()

    combined_activations = activations1
    combined_activations.extend(activations2)
    cka2.update_state(combined_activations)
    cka2_result = cka2.result().numpy()[:n_layers, -n_layers2:]
    assert (np.max(np.abs(cka2_result - cka1_result)) < 1e-5)

def get_CKA(n_layers, n_layers2, activations1, activations2):
    cka = MinibatchCKA(n_layers, n_layers2, across_models=True)
    cka.update_state_across_models(activations1, activations2)
    cka_result = cka.result().numpy()
    return cka_result

def get_r2_score(params_true, params_NN):
    r2_scores_params = []
    for i in range(params_true.shape[1]):
        r2_scores_params.append(
            r2_score(params_true[:, i], params_NN[:, i])
        )
    return r2_scores_params

def get_rmse_score(params_true, params_NN):
    rmse_scores_params = []
    for i in range(params_true.shape[1]):
        rmse_scores_params.append(
            mean_squared_error(params_true[:, i], params_NN[:, i], squared=False)  # squared=False means RMSE
        )
    return rmse_scores_params

# Power spectrum
def power_spectrum(delta, BoxSize=1000, vel_field=False, dimensional=2, MAS='CIC', axis=0):
    import Pk_library.Pk_library as PKL

    """Calculates the 2D power spectrum of a 2D field. It internally calculates the contrast field and then calculates the power spectrum.
    For velocity fields, the field is kept as it is.

    Args:
        delta (numpy.ndarray): Density slice (note: this is the field rather than the field contrast).
        BoxSize (float): Simulation box size.
    Returns:
        (numpy.ndarray, numpy.ndarray): The wavenumbers and power spectrum amplitudes.
    """
    delta = delta.astype(np.float32)
    if not vel_field:
        # Calculate contrast field.
        delta = delta / np.mean(delta, dtype=np.float64); delta = delta - 1.0

    threads = 2
    if dimensional == 2:
        Pk2D2 = PKL.Pk_plane(delta, BoxSize, MAS, threads)
        # get the attributes of the routine
        k2      = Pk2D2.k      #k in h/Mpc
        Pk2     = Pk2D2.Pk     #Pk in (Mpc/h)^2
    elif dimensional == 3:
        Pk2D2 = PKL.Pk(delta, BoxSize, axis, MAS, threads)
        # get the attributes of the routine
        k2      = Pk2D2.k3D     #k in h/Mpc
        Pk2     = Pk2D2.Pk[:, 0]     #Pk in (Mpc/h)^2

    return k2, Pk2

def smooth_3D_field(image, BoxSize=1000, R=50):
    import smoothing_library as SL

    field = image.astype(np.float32)
    grid    = field.shape[0]
    Filter  = 'Top-Hat'
    threads = 1
    #kmin    = 0  #h/Mpc
    #kmax    = 10 #h/Mpc

    # compute the filter in Fourier space
    W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
    # smooth the field
    field_smoothed = SL.field_smoothing(field, W_k, threads)

    return field_smoothed
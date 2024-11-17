import os
import numpy as np

# This is the folder which should contain a folder named `my_outputs_halo`
# and files of the format "halos_sim1003_LH_z0_grid64_masCIC.h5" inside this folder.
halo_dirname = './'

ANALYSE_BIAS_NOW = False
SAME_SIMS = True  # Whether the halo and DM density simulations exactly match.
if ANALYSE_BIAS_NOW and SAME_SIMS:
    # Analysis of the bias parameter
    import glob
    from utils import read_hdf5
    from utils import smooth_3D_field

    DEN_FIELD_DIRECTORY = 'my_outputs'

    dpath = os.path.join('.', f'{DEN_FIELD_DIRECTORY}', '*.h5')
    hpath = os.path.join(f'{halo_dirname}', f'{DEN_FIELD_DIRECTORY}_halo', '*.h5')

    all_bias_params = []
    dens = []
    halos = []
    for i, filename in enumerate(
        zip(
          sorted(glob.glob(dpath)),
          sorted(glob.glob(hpath))
        )
    ):
        # print(filename[0], filename[1])
        den, dparams = read_hdf5(filename[0], dataset_name='3D_density_field')
        halo, hparams = read_hdf5(filename[1], dataset_name='3D_halo_distribution')

        den = smooth_3D_field(den)
        halo = smooth_3D_field(halo)

        den_contrast = den/den.mean() - 1
        halo_contrast = halo/halo.mean() - 1

        dens.append(den_contrast)
        halos.append(halo_contrast)
    #     bias_params = (halo_contrast[np.where(den_contrast < 1)]/den_contrast[np.where(den_contrast < 1)])
        bias_params = (halo_contrast/den_contrast)

        # # Remove outliers.
        # bias_params = bias_params[(bias_params > np.quantile(bias_params, 0.01)) & (bias_params < np.quantile(bias_params, 0.99))]
        all_bias_params.append(np.median(bias_params))

    all_bias_params = np.array(all_bias_params)
    bias = np.mean(all_bias_params)

    print(f'Bias: {bias}')


############################################################
# Create transfer learning data
############################################################

# NOTE: use precomputed_mean, precomputed_stddev, precomputed_min_vals, and precomputed_max_vals only if you are using bias.
# Else it's better to use statistics of this new dataset for preprocessing.

#########################################################################################################

# The logic is that when the exact same sims are used for DM density and halo, we use the bias and also
# the precomputed statistics. Whereas if different simulations are used for DM and halo, we don't use the
# bias and also don't use the precomputed statistics. When same sims, the bias adds a positive value, thus
# log10(halo) does not give divide by zero error. When different sims, the same is handled by log(1+halo).

#########################################################################################################
    
num_sims = 2000
num_maps_per_projection_direction = 10
prefix = 'halos'

CREATE_DATASET_NOW = False
if CREATE_DATASET_NOW:
    prefix_original = ''
    # Load the mean, std, min_vals and max_vals into variables. This should be calculated from the pretraining dataset.
    MEAN = np.load(f'{prefix_original}_dataset_mean.npy')
    STD = np.load(f'{prefix_original}_dataset_std.npy')
    MIN_VALS = np.load(f'{prefix_original}_dataset_min_vals.npy')
    MAX_VALS = np.load(f'{prefix_original}_dataset_max_vals.npy')
    MEAN_DENSITIES = np.load(f'{prefix_original}_dataset_mean_densities.npy')
    print(MEAN, STD, MIN_VALS, MAX_VALS)

    if SAME_SIMS:
        command = [
            'python', 'create_data.py', '--num_sims', f'{num_sims}', '--train_frac', '0.8', '--test_frac', '0.1', '--seed', '42', '--path', f'{halo_dirname}/my_outputs_halo', '--grid_size', '256',
            '--num_maps_per_projection_direction', f'{num_maps_per_projection_direction}', '--prefix', prefix, '--dataset_name', '3D_halo_distribution', '--bias', f'{bias}',
            '--precomputed_mean', f'{MEAN}', '--precomputed_stddev', f'{STD}',
            '--precomputed_min_vals', f'{MIN_VALS[0]}', f'{MIN_VALS[1]}', f'{MIN_VALS[2]}', f'{MIN_VALS[3]}', f'{MIN_VALS[4]}',
            '--precomputed_max_vals', f'{MAX_VALS[0]}', f'{MAX_VALS[1]}', f'{MAX_VALS[2]}', f'{MAX_VALS[3]}', f'{MAX_VALS[4]}',
            '--smallest_sim_number', '0',
        ]
    else:  # don't use bias.'
        command = [
            'python', 'create_data.py', '--num_sims', f'{num_sims}', '--train_frac', '0.8', '--test_frac', '0.1', '--seed', '42', '--path', f'{halo_dirname}/my_outputs_halo', '--grid_size', '256',
            '--num_maps_per_projection_direction', f'{num_maps_per_projection_direction}', '--prefix', prefix, '--dataset_name', '3D_halo_distribution',
            '--smallest_sim_number', '1000', '--log_1_plus'
        ]
        #         --precomputed_mean {MEAN} --precomputed_stddev {STD} \
        #         --precomputed_min_vals {MIN_VALS[0]} {MIN_VALS[1]} {MIN_VALS[2]} {MIN_VALS[3]} {MIN_VALS[4]} \
        #         --precomputed_max_vals {MAX_VALS[0]} {MAX_VALS[1]} {MAX_VALS[2]} {MAX_VALS[3]} {MAX_VALS[4]} \

    import subprocess
    result = subprocess.run(command)
    print(result)
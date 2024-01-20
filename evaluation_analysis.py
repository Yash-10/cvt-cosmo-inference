import os
import glob
import gzip
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_rmse_score, unprocess_a_map
from model_dataset import model_o3_err
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


def post_test_analysis(
        params_true, params_NN, errors_NN, filenames,
        test_loader, params, num_sims, MEAN, STD, MEAN_DENSITIES, minimum, maximum,
        num_maps_per_projection_direction, hidden, dr, channels, fmodel,
        device='cpu', test_results_filename='test_results.csv', cka_filename='cka_matrix_pretrained_CNN_grid64_test.png'
):
    """This function is designed to make it easier to run inference in multiple experiments for easier comparison.

    Args:
        params_true (_type_): _description_
        params_NN (_type_): _description_
        errors_NN (_type_): _description_
        filenames (_type_): _description_
        test_loader (_type_): _description_
        params (_type_): _description_
        num_sims (_type_): _description_
        MEAN (_type_): _description_
        STD (_type_): _description_
        MEAN_DENSITIES (_type_): _description_
        num_maps_per_projection_direction (_type_): _description_
        hidden (_type_): _description_
        dr (_type_): _description_
        channels (_type_): _description_
        fmodel (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        test_results_filename (str, optional): _description_. Defaults to 'test_results.csv'.
        cka_filename (str, optional): _description_. Defaults to 'cka_matrix_pretrained_CNN_grid64_test.png'.

    Returns:
        _type_: _description_
    """
    import seaborn as sns
    sns.set_context("paper", font_scale = 2)
    sns.set_style('whitegrid')
    sns.set(style='ticks')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from utils import plot_results1, plot_results2, plot_results3, plot_std_sim, get_CKA

    # Calculate chi-squared score. See https://iopscience.iop.org/article/10.3847/1538-4357/acac7a
    assert len(params_true) == len(params_NN)
    def get_chi_square_score(params_true, params_NN, param_index):
        chi_square_score = (1 / len(params_true[:, param_index])) * (
            np.sum(
                    ((params_true[:, param_index] - params_NN[:, param_index]) ** 2) / (errors_NN[:, param_index] ** 2)
            )
        )
        return chi_square_score

    print('Chi-squared scores')
    print('Omega_m', get_chi_square_score(params_true, params_NN, 0))
    print('Omega_b', get_chi_square_score(params_true, params_NN, 1))
    print('h', get_chi_square_score(params_true, params_NN, 2))
    print('n_s', get_chi_square_score(params_true, params_NN, 3))
    print('sigma_8', get_chi_square_score(params_true, params_NN, 4))

    # Create a dataframe of results
    df = pd.DataFrame(np.hstack((np.expand_dims(filenames, 1), params_true, params_NN, errors_NN)))
    df.columns = ['filename'] + [f'params_true_{i}' for i in range(len(params))] + [f'params_NN_{i}' for i in range(len(params))] + [f'errors_NN_{i}' for i in range(len(params))]
    df.to_csv('test_results.csv')
    
    params_true2 = []
    averaged_params_NN = []
    averaged_errors_NN = []
    std_sim_NN = []

    for i in range(num_sims):
        df_subset = df[df['filename'].str.contains(f'_sim{i}_')]

        if df_subset.empty:  # This simulation was not in the test set
            continue

        p = [np.mean(df_subset[f'params_NN_{j}']) for j in range(len(params))]
        e = [np.mean(df_subset[f'errors_NN_{j}']) for j in range(len(params))]

        # Standard deviation of all point estimates for a single simulation.
        p_std = [np.std(df_subset[f'params_NN_{j}']) for j in range(len(params))]

        averaged_params_NN.append(p)
        averaged_errors_NN.append(e)
        std_sim_NN.append(p_std)
        params_true2.append(df_subset.iloc[0][[f'params_true_{k}' for k in range(len(params))]].tolist())

    params_true2 = np.vstack(params_true2)
    averaged_params_NN = np.vstack(averaged_params_NN)  # Averaged of model predictions for all 2D maps from a single simulation 3D cube. Ground-truth parameters for all maps of a simulation are the same.
    averaged_errors_NN = np.vstack(averaged_errors_NN)
    std_sim_NN = np.vstack(std_sim_NN)

    rmse_score = get_rmse_score(params_true, params_NN)
    sigma_bar = np.mean(errors_NN, axis=0)
    plot_results1(0, r'$\Omega_{\rm m}: $' + f'RMSE = {rmse_score[0]:.3f}, ' + r'$\bar{\sigma} = ' + f'{sigma_bar[0]:.3f}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results1(1, r'$\Omega_{\rm b}: $' + f'RMSE = {rmse_score[1]:.3f}, ' + r'$\bar{\sigma} = ' + f'{sigma_bar[1]:.3f}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results1(2, r'$h: $' + f'RMSE = {rmse_score[2]:.3f}, ' + r'$\bar{\sigma} = ' + f'{sigma_bar[2]:.3f}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results1(3, r'$n_s: $' + f'RMSE = {rmse_score[3]:.3f}, ' + r'$\bar{\sigma} = ' + f'{sigma_bar[3]:.3f}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results1(4, r'$\sigma_8: $' + f'RMSE = {rmse_score[4]:.3f}, ' + r'$\bar{\sigma} = ' + f'{sigma_bar[4]:.3f}$', params_true, params_NN, errors_NN, minimum, maximum)

    # This plots the predicted error on the parameter / predicted value of the parameter.
    # This can be used to mention the constraining accuracy: see Fig. 3 of https://arxiv.org/pdf/2109.09747.pdf
    # NOTE: One small caveat here is that this plot includes multiple maps from the same simulation, so the intrinsic differences in the error and parameter value will also be present here.
    plot_results3(0, r'$\Omega_{\rm m}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results3(1, r'$\Omega_{\rm b}$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results3(2, r'$h$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results3(3, r'$n_s$', params_true, params_NN, errors_NN, minimum, maximum)
    plot_results3(4, r'$\sigma_8$', params_true, params_NN, errors_NN, minimum, maximum)

    plot_results2(0, r'$\Omega_{\rm m}$', params_true2, averaged_params_NN, averaged_errors_NN, minimum, maximum)
    plot_results2(1, r'$\Omega_{\rm b}$', params_true2, averaged_params_NN, averaged_errors_NN, minimum, maximum)
    plot_results2(2, r'$h$', params_true2, averaged_params_NN, averaged_errors_NN, minimum, maximum)
    plot_results2(3, r'$n_s$', params_true2, averaged_params_NN, averaged_errors_NN, minimum, maximum)
    plot_results2(4, r'$\sigma_8$', params_true2, averaged_params_NN, averaged_errors_NN, minimum, maximum)

    # COV distribution. COV is calculated for each simulation, and its histogram is shown.
    plot_std_sim(0, r'$\Omega_{\rm m}$', std_sim_NN, averaged_params_NN)
    plot_std_sim(1, r'$\Omega_{\rm b}$', std_sim_NN, averaged_params_NN)
    plot_std_sim(2, r'$h$', std_sim_NN, averaged_params_NN)
    plot_std_sim(3, r'$n_s$', std_sim_NN, averaged_params_NN)
    plot_std_sim(4, r'$\sigma_8$', std_sim_NN, averaged_params_NN)

    print(f'Median COV for Omega_m: {np.median(std_sim_NN[:, 0]/averaged_params_NN[:, 0])}')
    print(f'Median COV for sigma_8: {np.median(std_sim_NN[:, 4]/averaged_params_NN[:, 4])}')

    df = pd.read_csv(test_results_filename)
    # test_sims = df['filename'].apply(lambda s:s.split('/')[1].split('_')[1]).unique()

    # WE SHOW ONLY A FEW EXAMPLES TO CHECK.
    counter = 0  # Counter to control how many cases to show. Set to zero, incremented after every iteration.
    for i in range(num_sims):
        files = glob.glob(f'/kaggle/working/test/*_sim{i}_*.npy.gz')
        if len(files) == 0:
            continue
        values = []
        for f in files:
            fg = gzip.GzipFile(f, 'r')
            den_2D = np.load(fg)
            den_2D = unprocess_a_map(den_2D, MEAN, STD, MEAN_DENSITIES[i], log_1_plus=False)
            values.append(den_2D.mean())

        df_subset = df[df['filename'].str.contains(f'_sim{i}_')]

        preds = []
        for param_index in [0, 4]:
            preds.append(
                df_subset[f'params_NN_{param_index}']
            )

        print(f'Simulation {i}')
        param_index = 0
        assert len(df_subset[f'params_true_{param_index}'].unique()) == 1  # For a single simulation, there must be only a single true parameter value.
        fig, ax = plt.subplots(1, 1)
        ax.scatter(values, preds[0])
        ax.set_xlabel('Mean density of 2D maps from a single simulation')
        ax.set_ylabel('Predicted parameter value')
        ax.axhline(y=df_subset[f'params_true_{param_index}'].unique()[0], linestyle='--', c='black')
        ax.set_title(r'$\Omega_m$')
        plt.show()

        param_index = 4
        assert len(df_subset[f'params_true_{param_index}'].unique()) == 1  # For a single simulation, there must be only a single true parameter value.
        fig, ax = plt.subplots(1, 1)
        ax.scatter(values, preds[1])  # Note [1] instead of [4] because preds contains only for Omega_m and sigma_8.
        ax.set_xlabel('Mean density of 2D maps from a single simulation')
        ax.set_ylabel('Predicted parameter value')
        ax.axhline(y=df_subset[f'params_true_{param_index}'].unique()[0], linestyle='--', c='black')
        ax.set_title(r'$\sigma_8$')
        plt.show()

        counter += 1

        if counter == 3:
            break


    # Redoing the above plot, but showing the coefficient of variation on y-axis
    # WE SHOW ONLY A FEW EXAMPLES TO CHECK.
    counter = 0  # Counter to control how many cases to show. Set to zero, incremented after every iteration.
    for i in range(num_sims):
        files = glob.glob(f'/kaggle/working/test/*_sim{i}_*.npy.gz')
        if len(files) == 0:
            continue
        values = []
        for f in files:
            fg = gzip.GzipFile(f, 'r')
            den_2D = np.load(fg)
            den_2D = unprocess_a_map(den_2D, MEAN, STD, MEAN_DENSITIES[i], log_1_plus=False)
            values.append(den_2D.mean())

        df_subset = df[df['filename'].str.contains(f'_sim{i}_')]

        covs = []
        for param_index in [0, 4]:
            covs.append(
                df_subset[f'errors_NN_{param_index}']/df_subset[f'params_NN_{param_index}']
            )

        print(f'Simulation {i}')
        print(len(values), covs[0].shape, covs[1].shape)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(values, covs[0])
        ax.set_xlabel('Mean density of 2D maps from a single simulation')
        ax.set_ylabel('Coefficient of variation')
        ax.set_title(r'$\Omega_m$')
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.scatter(values, covs[1])  # Note [1] instead of [4] because cov contains only for Omega_m and sigma_8.
        ax.set_xlabel('Mean density of 2D maps from a single simulation')
        ax.set_ylabel('Coefficient of variation')
        ax.set_title(r'$\sigma_8$')
        plt.show()

        counter += 1

        if counter == 3:
            break

    # Now doing the same test as above, but testing the variance in estimates across simulations.
    # The variance here must be much larger than the variance of results on a single simulation.
    params_true2 = []
    averaged_params_NN = []
    averaged_errors_NN = []
    std_sim_NN = []
    counter = 0

    for i in range(num_maps_per_projection_direction*3):  # Total no. of 2d maps from a single 3d cube.
        for direction in ['X', 'Y', 'Z']:
            df_subset = df[df['filename'].str.contains(f'_{direction}{i}_')]

            if df_subset.empty:  # This 2d map was not in the test set for any test simulation.
                continue

            p = [np.mean(df_subset[f'params_NN_{j}']) for j in range(len(params))]
            e = [np.mean(df_subset[f'errors_NN_{j}']) for j in range(len(params))]

            for ss in range(len(params)):
                # Each value must be from a different simulation, so no overlap must be there.
                assert np.unique(df_subset[f'params_true_{ss}']).shape == df_subset[f'params_true_{ss}'].shape

            # Standard deviation of all point estimates for a single simulation.
            p_std = [np.std(df_subset[f'params_NN_{j}']) for j in range(len(params))]

            averaged_params_NN.append(p)
            averaged_errors_NN.append(e)
            std_sim_NN.append(p_std)
            params_true2.append(df_subset.iloc[0][[f'params_true_{k}' for k in range(len(params))]].tolist())
            counter += 1

    assert counter == (num_maps_per_projection_direction * 3) * 3

    params_true2 = np.vstack(params_true2)
    averaged_params_NN = np.vstack(averaged_params_NN)
    averaged_errors_NN = np.vstack(averaged_errors_NN)
    std_sim_NN = np.vstack(std_sim_NN)

    # We use the same function as the above test in the above cell, but here the variables themselves are changed.
    plot_std_sim(0, r'$\Omega_{\rm m}$', std_sim_NN, averaged_params_NN)
    plot_std_sim(1, r'$\Omega_{\rm b}$', std_sim_NN, averaged_params_NN)
    plot_std_sim(2, r'$h$', std_sim_NN, averaged_params_NN)
    plot_std_sim(3, r'$n_s$', std_sim_NN, averaged_params_NN)
    plot_std_sim(4, r'$\sigma_8$', std_sim_NN, averaged_params_NN)

    # Load model.
    model = model_o3_err(hidden, dr, channels)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    # load the weights in case they exists
    if os.path.exists(fmodel):
        model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
        print('Weights loaded')
    
    data_batch = []
    for i, (x, y, _) in enumerate(test_loader):
        if i == 1:
            data_batch.append(x)
            break

    x = torch.vstack(data_batch)
    x = x.to(device)

    mid_getter = MidGetter(model, return_layers={'LeakyReLU': 'LeakyReLU'}, keep_output=True)
    mid_outputs = mid_getter(x)
    print(len(mid_outputs[0]['LeakyReLU']))  # Will be 6.

    data_batch = []
    for i, (x, y, _) in enumerate(test_loader):
        data_batch.append(x)

    x = torch.vstack(data_batch)
    x = x.to(device)

    with torch.no_grad():
        mid_getter = MidGetter(model, return_layers={'LeakyReLU': 'LeakyReLU'}, keep_output=True)
        mid_outputs1 = mid_getter(x)
        mid_outputs2 = mid_getter(x)

        intermediate_outputs_A = mid_outputs1[0]['LeakyReLU']
        intermediate_outputs_B = mid_outputs2[0]['LeakyReLU']

        intermediate_outputs_A = [o.cpu() for o in intermediate_outputs_A]
        intermediate_outputs_B = [ob.cpu() for ob in intermediate_outputs_B]

        sim = get_CKA(n_layers=6, n_layers2=6, activations1=intermediate_outputs_A, activations2=intermediate_outputs_B)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(sim, vmin=0, vmax=1)
    ax.axes.invert_yaxis()

    ax.set_xlabel('Layers 1')
    ax.set_ylabel('Layers 2')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical');
    plt.savefig(cka_filename, bbox_inches='tight', dpi=200)
    plt.show()


def gradcam_evaluation(test_loader, fmodel, hidden, dr, channels, device='cpu'):
    from grad_cam_interpret import GradCAMRegressor
    import torch.nn.functional as F

    images, labels, _ = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Example usage:
    # Assuming 'model' is your regression model and 'target_layer' is the layer you want to visualize
    for img_idx in [5, 10, 15]:
        # Only select one image
        images_ = images[img_idx, :].unsqueeze(0)
        print(images_.shape)
        for index in [0, 4]:
            if index == 0:
                param = r'$\Omega_m$'
            elif index == 4:
                param = r'$\sigma_8$'

            #Load model.
            model = model_o3_err(hidden, dr, channels)
            model.to(device=device)
            # load the weights in case they exists
            if os.path.exists(fmodel):
                model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
                print('Weights loaded')

            gradcam_regressor = GradCAMRegressor(model, target_layer=model.B21, ground_truth_param_value=labels[:, index][img_idx], index=index)

            # Visualize Grad-CAM for the entire image in a regression context
            gradcam = gradcam_regressor.generate_gradcam(images_)

            # Remove hooks after usage
            gradcam_regressor.remove_hooks()

            fig, ax = plt.subplots(1, 3, figsize=(10, 6))
            show_image = images_.squeeze().cpu().detach().numpy()
            ax[0].imshow(show_image)
            ax[1].imshow(gradcam.cpu().detach().numpy())

            gradcam = gradcam.unsqueeze(0).unsqueeze(0)

            # Resize Grad-CAM to match the input image size
            gradcam = F.interpolate(gradcam, size=show_image.shape, mode='bilinear', align_corners=False)
            # Convert to numpy array for visualization
            gradcam = gradcam.squeeze().cpu().detach().numpy()
            # Normalize for visualization
            gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam) + 1e-8)
            original_image = images_[0].permute(1, 2, 0).detach().cpu().numpy()
            original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image) + 1e-8)
            ax[2].imshow(original_image)
            # Overlay Grad-CAM on the original image
            ax[2].imshow(gradcam, cmap='jet', alpha=0.3, interpolation='bilinear')
            ax[0].set_title(param)
            plt.show()
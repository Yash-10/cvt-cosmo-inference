# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Main script to run the testing. The trained model from
#              `main_train.py` is loaded and used for testing.
# ***************************************************************************

import numpy as np

from train_val_test_boilerplate import test
from model.vit import ViT
from model.cnn import CNN

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils import CustomImageDataset

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from utils import extract_model_params

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Parameters
test_dir = f'./my_outputs/test'

#pretrained_dir = f'./saved_models/lightning_logs_csv/version_6'
#pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=27-step=84000.ckpt'

pretrained_dir = f'./saved_models/lightning_logs_csv/version_7'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=29-step=90000.ckpt'


# Load statistics
prefix = ''
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print(MEAN, STD, MIN_VALS, MAX_VALS)

# Load test data
batch_size = 64
test_dataset = CustomImageDataset(test_dir, normalized_cosmo_params_path=f'{test_dir}/test_normalized_params.csv', transform=None)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

# Load trained model.
print("####### loading model #########")
filename = f'{pretrained_dir}/hparams.yaml'
model_kwargs = extract_model_params(filename, 'model_kwargs')

print(model_kwargs)
model = ViT.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
#model = CNN.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)


params = [0,1,2,3,4]
g = params
h = [5+i for i in g]

params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=MIN_VALS, maximum=MAX_VALS)

num_sims = 2000
num_maps_per_projection_direction = 10 
from evaluation_analysis import post_test_analysis, get_cka
post_test_analysis(
    params_true, params_NN, errors_NN, filenames,
    params, num_sims, MEAN, STD, MEAN_DENSITIES, MIN_VALS, MAX_VALS,
    num_maps_per_projection_direction, test_results_filename='test_results.csv',
    smallest_sim_number=0
)

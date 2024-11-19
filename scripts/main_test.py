# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Main script to run the testing. The trained model from
#              `main_train.py` is loaded and used for testing.
# ***************************************************************************
import sys
import time
import numpy as np

from train_val_test_boilerplate import test
from model.vit import ViT, ViT_FineTune, ViT_FineTune_CvT
from model.cnn import CNN, CNN_FineTune

import torch
from torch.utils.data import DataLoader
import torchvision
from utils import CustomImageDataset, MyRotationTransform, MyFilterSmallValues
from utils import extract_model_params

import wandb
import pandas as pd

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Parameters
FILTERING = False

## halo from scratch num_sim = 100 

prefix = ''
base_dir = f'./my_outputs'

pretrained_dir = './saved_models_halo/lightning_logs_csv/version_28'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=21-step=2200.ckpt'

df = pd.read_csv("./pretrained_filename.txt", delimiter=' ', header=None)
if pretrained_filename in df[0].values:
    name = df[df[0] == pretrained_filename][1].values[0]
else:    
    import uuid
    random_string = str(uuid.uuid4())[:8]
    name = f'{prefix}_{random_string}'
print(f"name: {name}")
if "filtered" in name:
    FILTERING = True


import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config['params'] #[0,1,2,3,4]           
g = params
h = [5+i for i in g]
project_name = config['project_name']
model_name = config['model_name']

# Load statistics
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')[g]
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')[g]
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print(MEAN, STD, MIN_VALS, MAX_VALS)

# Load test data
transform = None
if FILTERING:
    print("!!! Filtering small values !!!")
    transform = torchvision.transforms.Compose([
        MyRotationTransform(angles=[0]),
        MyFilterSmallValues(threshold=0)
    ])

batch_size = 64
test_dataset = CustomImageDataset(f'{base_dir}/test', normalized_cosmo_params_path=f'{base_dir}/test/test_normalized_params.csv', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

# Load trained model.
print("####### loading model #########")
filename = f'{pretrained_dir}/hparams.yaml'
model_kwargs = extract_model_params(filename, 'model_kwargs')

print(model_kwargs)
if 'ViT' in model_name or model_name == 'CvT':
    if "transfer" in name:
        if "CvT" in name:
            model = ViT_FineTune_CvT.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
        else:
            model = ViT_FineTune.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
    else:
        model = ViT.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
elif model_name == 'CNN':
    if "transfer" in name:
        model = CNN_FineTune.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
    else:
        model = CNN.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
else:
    raise ValueError("Model not supported")

time0 = time.time()
params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=MIN_VALS, maximum=MAX_VALS)
time_taken_test = time.time() - time0
print(f'Time taken for testing: {time_taken_test:.2f} seconds')
with open("time_for_test.txt", "a") as f:
    f.write(f'{name} {time_taken_test:.2f}\n')

num_sims = 2000
num_maps_per_projection_direction = 10 


wandb.init(project=project_name, name=name, mode="online")
from evaluation_analysis import post_test_analysis
full_name = name.replace("/", "_")
post_test_analysis(
    params_true, params_NN, errors_NN, filenames,
    params, num_sims, MEAN, STD, MEAN_DENSITIES, MIN_VALS, MAX_VALS,
    num_maps_per_projection_direction, test_results_filename=f'test_results_{full_name}.csv',
    smallest_sim_number=0
)


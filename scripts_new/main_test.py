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

prefix = 'dens128'
base_dir = f'./my_outputs_128'
pretrained_dir = f'./saved_models_dens128/lightning_logs_csv/version_0'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=27-step=28000.ckpt'

prefix = ''
base_dir = f'./my_outputs'
pretrained_dir = f'./saved_models/lightning_logs_csv/version_0'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=25-step=26000.ckpt'

prefix = ''
base_dir = f'./my_outputs'
pretrained_dir = f'./saved_models/lightning_logs_csv/version_5'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=23-step=2400.ckpt'

prefix = ''
base_dir = f'./my_outputs'
pretrained_dir = f'./saved_models/lightning_logs_csv/version_4'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=28-step=29000.ckpt'

prefix = 'halos'
base_dir = f'./my_outputs_halo'
pretrained_dir = f'./saved_models_halos/lightning_logs_csv/version_0'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=20-step=2100.ckpt'

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
    model = ViT.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
elif model_name == 'CNN':
    model = CNN.load_from_checkpoint(pretrained_filename, model_kwargs=model_kwargs, minimum=MIN_VALS, maximum=MAX_VALS)
else:
    raise ValueError("Model not supported")

params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=MIN_VALS, maximum=MAX_VALS)

num_sims = 2000
num_maps_per_projection_direction = 10 

wandb.init(project=project_name, name=name, mode="online")
from evaluation_analysis import post_test_analysis
post_test_analysis(
    params_true, params_NN, errors_NN, filenames,
    params, num_sims, MEAN, STD, MEAN_DENSITIES, MIN_VALS, MAX_VALS,
    num_maps_per_projection_direction, test_results_filename='test_results.csv',
    smallest_sim_number=0
)

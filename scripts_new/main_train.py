# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Main script to run the training. All hyperparameters/variables
#              are hardcoded in this script.
# ***************************************************************************


## KM added
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

## Standard libraries
import os
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch

## Torchvision
import torchvision
# from torchvision.datasets import CIFAR10

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model.vit import ViT, load_model_for_torch_summary
from model.cnn import CNN
from utils import MyRotationTransform

# CONSTANTS
base_dir = './my_outputs'
prefix = ''

base_dir = './my_outputs_halo'
prefix = 'halo'

# Path to the folder where the pretrained models are saved
if prefix == '':
    CHECKPOINT_PATH = "./saved_models"
else:
    CHECKPOINT_PATH = f"./saved_models_{prefix}"

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Setting the seed
SEED = 42
pl.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# SET HYPERPARAMETERS.
# optimizer parameters
beta1 = 0.5
beta2 = 0.999

batch_size = 16
lr         = 5e-6
wd         = 1e-5  # value of weight decay
dr         = 0.2
epochs     = 30    # number of epochs to train the network
channels        = 1                #we only consider here 1 field

GRID_SIZE = 256
num_maps_per_projection_direction = 10
num_sims = 2000

image_size = GRID_SIZE
patch_size = 8

model_name = "ViT"
model_kwargs = {
    'image_size': image_size,
    'embed_dim': 1024,  # For the MLP.
    'hidden_dim': 64, # this is the projection_dim in terms of the variables defined in https://keras.io/examples/vision/image_classification_with_vision_transformer/.
    'num_heads': 4,
    'num_layers': 8,
    'patch_size': patch_size,
    'num_channels': channels,
    'num_patches': (image_size // patch_size) ** 2,
    'num_classes': 10,
    'dropout': dr
}

"""
model_name = "CNN"
model_kwargs = {
    'image_size': image_size,
    'hidden': 8,
    'dr': dr,
    'channels': channels
}
"""

"""
## For CrossViT ##
model_name = "CrossViT"
model_kwargs = dict(
    image_size = image_size,
    num_classes = 10,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 4,       # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 16,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1,
    num_channels = channels
)
"""

# STEP 1: Create data for deep learning
create_dataset_now = False
if create_dataset_now:
    print("creating dataset")
    import subprocess
    command = [
        'python', 'create_data.py', '--num_sims', f'{num_sims}', '--train_frac', '0.8', '--test_frac', '0.1',
        '--seed', f'{SEED}', '--path', './my_outputs', '--grid_size', f'{GRID_SIZE}',
        '--num_maps_per_projection_direction', f'{num_maps_per_projection_direction}', '--prefix', '',
        '--smallest_sim_number', '0', '--log_1_plus'
    ]
    result = subprocess.run(command)
    print(result)

# Store the mean, std, min_vals and max_vals into variables
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print(MEAN, STD, MIN_VALS, MAX_VALS)

# STEP 2: Create dataloader (with data augmentations).
from torch.utils.data import DataLoader
from utils import CustomImageDataset
import torchvision.transforms.functional as TF

import random

transform = torchvision.transforms.Compose([
    MyRotationTransform(angles=[90, 180, 270]),
    #v2.ToDtype(torch.float32),#, scale=False),
])

train_dataset = CustomImageDataset(f'{base_dir}/train', normalized_cosmo_params_path=f'{base_dir}/train/train_normalized_params.csv', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

val_dataset = CustomImageDataset(f'{base_dir}/val', normalized_cosmo_params_path=f'{base_dir}/val/val_normalized_params.csv', transform=None)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
# num_workers=3 was suggested by PyTorch Lightning while running the `train_model` function on Kaggle.

# STEP 3: Define the ViT model

import torch

# STEP 4: Define training loop
logger_csv = pl.loggers.CSVLogger(CHECKPOINT_PATH, name="lightning_logs_csv")

def train_model(model_name, **kwargs):
    # See https://lightning.ai/docs/pytorch/2.1.3/common/trainer.html#reproducibility
    pl.seed_everything(SEED, workers=True) # To be reproducible

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=epochs,
                         logger=[logger_csv],
                         log_every_n_steps=int(len(train_dataset)/batch_size),
#                          progress_bar_refresh_rate=50,  # recommended for Kaggle/Colab here: https://www.youtube.com/watch?v=-XakoRiMYCg
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                             LearningRateMonitor("epoch")
                         ],
#                          deterministic=True
                        )
#     trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
#     trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        if model_name == "ViT":
            model = ViT.load_from_checkpoint(pretrained_filename)
        elif model_name == "CNN":
            model = CNN.load_from_checkpoint(pretrained_filename)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # Automatically loads the model with the saved hyperparameters
    else:
        if model_name == "ViT":
            model = ViT(**kwargs)
        elif model_name == "CNN":
            model = CNN(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        trainer.fit(model, train_loader, val_loader)
        # Since we use every_n_epochs=None, by default the model is checkpointed after each epoch and the best is selected.
        
        if model_name == "ViT":
            model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
        elif model_name == "CNN":
            model = CNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_loss"]}

    return model, result, trainer.checkpoint_callback.best_model_path

# Check model forward pass works without errors
#from torchsummary import summary
#model_torch_summary = load_model_for_torch_summary(model_name, model_kwargs=model_kwargs)
#print(summary(model_torch_summary.to(device), (1, image_size, image_size)))


# STEP 5: Run training
model, results, PRETRAINED_FILENAME = train_model(
    model_name, model_kwargs=model_kwargs,
    lr=lr, wd=wd, beta1=beta1, beta2=beta2,
    minimum=MIN_VALS, maximum=MAX_VALS
)
print("ViT results", results)
model.to(device)
print(f'Pretrained file name: {PRETRAINED_FILENAME}')

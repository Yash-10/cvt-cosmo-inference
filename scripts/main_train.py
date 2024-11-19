# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Main script to run the training. All hyperparameters/variables
#              are hardcoded in this script.
# ***************************************************************************

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

## Standard libraries
import os
import time
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

from model.vit import ViT
from model.cnn import CNN

# frequently changed variables

FILTERING = False
CROP = False

## dens -> halo
base_dir = './my_outputs'
prefix = ''
image_size = 256
num_sims = 2000

#base_dir = './my_outputs_halo'
#prefix = 'halos'
#image_size = 256
#num_sims = 200


epochs = 30

num_sims_train = int(0.8 * num_sims)
num_sims_val = int(0.1 * num_sims)


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
beta1 = 0.5
beta2 = 0.999

batch_size = 16
lr         = 5e-6
wd         = 1e-5  # value of weight decay
dr         = 0
channels        = 1                #we only consider here 1 field

num_maps_per_projection_direction = 10


import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config['params'] #[0,1,2,3,4]           
g = params
h = [5+i for i in g]

project_name = config['project_name']
model_name = config['model_name']

if model_name == "SimpleViT":
    patch_size = 8
    model_kwargs = {
        'image_size': image_size,
        'embed_dim': 1024,  # For the MLP.
        'hidden_dim': 64, # this is the projection_dim in terms of the variables defined in https://keras.io/examples/vision/image_classification_with_vision_transformer/.
        'num_heads': 4,
        'num_layers': 8,
        'patch_size': patch_size,
        'num_channels': channels,
        'num_patches': (image_size // patch_size) ** 2,
        'num_classes': len(g) + len(h),
        'dropout': dr
    }
elif model_name == "CvT":
    model_kwargs = {
        'num_classes': len(g) + len(h),
        'dr': dr
    }
elif model_name == "CNN":
    model_kwargs = {
        'num_classes': len(g) + len(h),
        'image_size': image_size,
        'hidden': 8,
        'dr': dr,
        'channels': channels
    }
else:
    raise ValueError(f"Unknown model name: {model_name}")


print("model_name:", model_name)
print("model_kwargs:", model_kwargs)
print("parameters to be inferred:", params)

# Store the mean, std, min_vals and max_vals into variables
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')[g]
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')[g]
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print("MEAN, STD, MIN_VALS, MAX_VALS:", MEAN, STD, MIN_VALS, MAX_VALS)

# Create dataloader (with data augmentations).
from torch.utils.data import DataLoader
from utils import CustomImageDataset, MyRotationTransform, MyFilterSmallValues

transform = torchvision.transforms.Compose([
    MyRotationTransform(angles=[90, 180, 270]),
    #v2.ToDtype(torch.float32),#, scale=False),
])
val_transform = None

if FILTERING:
    transform = torchvision.transforms.Compose([
        MyRotationTransform(angles=[90, 180, 270]),
        MyFilterSmallValues(threshold=0)
    ])
    val_transform = torchvision.transforms.Compose([
        MyRotationTransform(angles=[0]),
        MyFilterSmallValues(threshold=0)
    ])
if CROP:
    transform = torchvision.transforms.Compose([
        MyRotationTransform(angles=[90, 180, 270]),
        torchvision.transforms.RandomCrop(image_size),
    ])
    val_transform = torchvision.transforms.Compose([
        MyRotationTransform(angles=[0]),
        torchvision.transforms.RandomCrop(image_size),
    ])

train_dataset = CustomImageDataset(f'{base_dir}/train', normalized_cosmo_params_path=f'{base_dir}/train/train_normalized_params.csv', transform=transform, nmax=num_sims_train*num_maps_per_projection_direction)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

val_dataset = CustomImageDataset(f'{base_dir}/val', normalized_cosmo_params_path=f'{base_dir}/val/val_normalized_params.csv', transform=val_transform, nmax=num_sims_val*num_maps_per_projection_direction)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
# num_workers=3 was suggested by PyTorch Lightning while running the `train_model` function on Kaggle.


# Define training loop
from pytorch_lightning.loggers import WandbLogger
logger_csv = pl.loggers.CSVLogger(CHECKPOINT_PATH, name="lightning_logs_csv")
WANDB_PROJECT_NAME = project_name
WANDB_RUN_NAME = f'ViT_train_{prefix}_num_sims-{num_sims}_batchsize-{batch_size}_lr-{lr}_epochs-{epochs}_wd-{wd}_dr-{dr}'
if FILTERING:
    WANDB_RUN_NAME += '_filtered'
if CROP:
    WANDB_RUN_NAME += '_cropped'
wandb_logger = WandbLogger(name=WANDB_RUN_NAME, project=WANDB_PROJECT_NAME)
wandb_logger.experiment.config.update({"batch_size": batch_size, "epochs": epochs})

print("WANDB_PROJECT_NAME:", WANDB_PROJECT_NAME)
print("WANDB_RUN_NAME:", WANDB_RUN_NAME)

def train_model(model_name, **kwargs):
    # See https://lightning.ai/docs/pytorch/2.1.3/common/trainer.html#reproducibility
    pl.seed_everything(SEED, workers=True) # To be reproducible

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=epochs,
                         logger=[logger_csv, wandb_logger],
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

    if "ViT" in model_name or model_name == "CvT":
        model = ViT(**kwargs)
    elif model_name == "CNN":
        model = CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    trainer.fit(model, train_loader, val_loader)
    # Since we use every_n_epochs=None, by default the model is checkpointed after each epoch and the best is selected.
    
    if "ViT" in model_name or model_name == "CvT":
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    elif model_name == "CNN":
        model = CNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_loss"]}

    return model, result, trainer.checkpoint_callback.best_model_path


# Run training
time0 = time.time()
model, results, PRETRAINED_FILENAME = train_model(
    model_name, model_kwargs=model_kwargs,
    lr=lr, wd=wd, beta1=beta1, beta2=beta2,
    minimum=MIN_VALS, maximum=MAX_VALS
)
time_taken = time.time() - time0
print("ViT results", results)
print(f'Pretrained file name: {PRETRAINED_FILENAME}')
print(f'Time taken: {time_taken:.2f} seconds')

TEST_NOW = True
time_taken_test = 0
if TEST_NOW:
    model.to(device)

    from train_val_test_boilerplate import test
    test_results_filename = f'test_results_{model_name}_{prefix}_num_sims{num_sims}_epoch{epochs}.csv'
    
    # Load test data
    batch_size = 64
    test_dataset = CustomImageDataset(f'{base_dir}/test', normalized_cosmo_params_path=f'{base_dir}/test/test_normalized_params.csv', transform=val_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    time0 = time.time()
    params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=MIN_VALS, maximum=MAX_VALS)
    time_taken_test = time.time() - time0
    print(f'Time taken for testing: {time_taken_test:.2f} seconds')

    import wandb
    wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, mode="online")

    from evaluation_analysis import post_test_analysis
    num_sims = 2000
    num_maps_per_projection_direction = 10 
    post_test_analysis(
        params_true, params_NN, errors_NN, filenames,
        params, num_sims, MEAN, STD, MEAN_DENSITIES, MIN_VALS, MAX_VALS,
        num_maps_per_projection_direction, test_results_filename=test_results_filename,
        smallest_sim_number=0
    )

with open("pretrained_filename.txt", "a") as f:
    f.write(f'{PRETRAINED_FILENAME} {WANDB_PROJECT_NAME}/{WANDB_RUN_NAME} {time_taken:.2f} {time_taken_test:.2f}\n')
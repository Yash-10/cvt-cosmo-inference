# ***************************************************************************
# Author: Yash Gondhalekar  Last updated: May 2024.

# Description: Main script to run transfer learning on halo data assuming the
#              model is trained on DM density.
#              All hyperparameters/variables are hardcoded in this script.
# ***** NOTE: This script finetunes the ViT model. *****
# ***************************************************************************

## KM added
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
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()


## PyTorch
import torch

## Torchvision
import torchvision
# from torchvision.datasets import CIFAR10

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from utils import extract_model_params
from model.vit import ViT_FineTune, ViT_FineTune_CvT
from model.cnn import CNN_FineTune

DR_NEW = 0
FREEZE_LAYERS = False  # If True, all layers except Linear in to_patch_embedding are frozen

### dens -> halo
base_dir = './my_outputs_halo'
prefix = 'halos'
CHECKPOINT_PATH = "./saved_models_halo"

# CvT mse

### Cosmo-parameter-inference-mse-CNN/ViT_train__num_sims-2000_batchsize-16_lr-5e-06_epochs-30_wd-1e-05_dr-0
pretrained_dir = "./saved_models/lightning_logs_csv/version_42"
pretrained_filename = f"{pretrained_dir}/checkpoints/epoch=29-step=30000.ckpt"

name_opt = ''

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# SET HYPERPARAMETERS
batch_size = 16
num_maps_per_projection_direction = 10
num_sims = 200
num_sims_train = int(0.8 * num_sims)
num_sims_val = int(0.1 * num_sims)

epochs = 30

# Setting the seed
SEED = 42
pl.seed_everything(SEED)

# Load other setting from config.yaml
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config['params'] #[0,1,2,3,4]           
g = params
h = [5+i for i in g]
project_name = config['project_name']
model_name = config['model_name']
print(config)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


## Load pretrained paramters
filename = f'{pretrained_dir}/hparams.yaml'
model_kwargs = extract_model_params(filename, 'model_kwargs')
lr = extract_model_params(filename, 'lr')
beta1 = extract_model_params(filename, 'beta1')
beta2 = extract_model_params(filename, 'beta2')
wd = extract_model_params(filename, 'wd')

if DR_NEW is not None:
    model_kwargs['dr'] = DR_NEW
    name_opt += f'_dr{DR_NEW}'
    
name_opt += '_frozen_both' if FREEZE_LAYERS else ''

print("Model name:", model_name)
print("Model kwargs:", model_kwargs)
print("Parameters to be inferred:", params)

# Load the mean, std, min_vals and max_vals into variables
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')[g]
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')[g]
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print(MEAN, STD, MIN_VALS, MAX_VALS)


# Create dataloader
from torch.utils.data import DataLoader
from utils import CustomImageDataset
from utils import MyRotationTransform

transform = torchvision.transforms.Compose([
    MyRotationTransform(angles=[90, 180, 270]),
])

train_dataset = CustomImageDataset(f'{base_dir}/train', normalized_cosmo_params_path=f'{base_dir}/train/train_normalized_params.csv', transform=transform, nmax=num_sims_train*num_maps_per_projection_direction)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

val_dataset = CustomImageDataset(f'{base_dir}/val', normalized_cosmo_params_path=f'{base_dir}/val/val_normalized_params.csv', transform=None, nmax=num_sims_val*num_maps_per_projection_direction)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

test_dataset = CustomImageDataset(f'{base_dir}/test', normalized_cosmo_params_path=f'{base_dir}/test/test_normalized_params.csv', transform=None) # I don't limit the number of test data.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

# Loggers
from pytorch_lightning.loggers import WandbLogger
logger_csv = pl.loggers.CSVLogger(CHECKPOINT_PATH, name="lightning_logs_csv")
WANDB_PROJECT_NAME = project_name
WANDB_RUN_NAME = f'ViT_transfer_{prefix}_num_sims-{num_sims}_batchsize-{batch_size}_lr-{lr}_epochs-{epochs}_wd-{wd}{name_opt}'
wandb_logger = WandbLogger(name=WANDB_RUN_NAME, project=WANDB_PROJECT_NAME)
wandb_logger.experiment.config.update({"batch_size": batch_size, "epochs": epochs})

print("WANDB_PROJECT_NAME:", WANDB_PROJECT_NAME)
print("WANDB_RUN_NAME:", WANDB_RUN_NAME)

# Define finetuning function
def finetune_model(pretrained_filename, model_name, **kwargs):
    pl.seed_everything(SEED, workers=True) # To be reproducable
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=epochs,
                         logger=[logger_csv, wandb_logger],
#                          progress_bar_refresh_rate=50,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                    LearningRateMonitor("epoch")],
#                          deterministic=True
                        )
#     trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
#     trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it before fine-tuning. If no, raise an error.
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}. This will be used to load the model.")
        print(kwargs["model_kwargs"])
        
        if model_name == "CvT":
            model = ViT_FineTune_CvT(pretrained_filename, **kwargs) 
        elif model_name == "SimpleViT":
            model = ViT_FineTune(pretrained_filename, **kwargs)
        elif model_name == "CNN":
            model = CNN_FineTune(pretrained_filename, **kwargs) 
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # Automatically loads the model with the saved hyperparameters
    else:
        raise ValueError(f"pretrained file {pretrained_filename} does not exist or is not file.")
        
    # After loading the pretrained model, finetune it.
    trainer.fit(model, train_loader, val_loader)

    if model_name == "CvT":
        model = ViT_FineTune_CvT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    elif model_name == "SimpleViT":
        model = ViT_FineTune.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    elif model_name == "CNN":
        model = CNN_FineTune.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_loss"], "val": val_result[0]["test_loss"]}

    return model, result, trainer.checkpoint_callback.best_model_path

## Finetune the model
time0 = time.time()
model, results, FINETUNED_FILENAME = finetune_model(
    pretrained_filename, model_name=model_name, model_kwargs=model_kwargs,
    lr=lr, wd=wd, beta1=beta1, beta2=beta2,
    minimum=MIN_VALS, maximum=MAX_VALS, freeze_layers=FREEZE_LAYERS
)
time_taken = time.time() - time0

print("ViT_FineTune results", results)
print(f'Fine-tuned file name: {FINETUNED_FILENAME}')
print(f'Time taken: {time_taken:.2f} seconds')


#################################################
# Now test the finetuned model.
#################################################

model.to(device)

from train_val_test_boilerplate import test
test_results_filename = f'test_results_transfer_learning_{model_name}_{prefix}_num_sims{num_sims}_epoch{epochs}_freeze{FREEZE_LAYERS}.csv'

# Below values calculated during data preparation. See above.
time0 = time.time()
params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=MIN_VALS, maximum=MAX_VALS)
time_taken_test = time.time() - time0
print(f'Time taken for testing: {time_taken_test:.2f} seconds')

import wandb
wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, mode="online")

from evaluation_analysis import post_test_analysis
num_sims = 2000 # Data up to this id is used for test
post_test_analysis(
    params_true, params_NN, errors_NN, filenames,
    params, num_sims, MEAN, STD, MEAN_DENSITIES, MIN_VALS, MAX_VALS,
    num_maps_per_projection_direction,
    test_results_filename=test_results_filename,
    smallest_sim_number=0 
)

with open("pretrained_filename.txt", "a") as f:
    f.write(f'{FINETUNED_FILENAME} {WANDB_PROJECT_NAME}/{WANDB_RUN_NAME} {time_taken:.2f} {time_taken_test:.2f}\n')


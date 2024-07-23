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
from model.vit import ViT_FineTune, load_model_for_torch_summary
from model.cnn import CNN_FineTune

# CONSTANTS
base_dir = './my_outputs_halo'
prefix = 'halos'

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models"
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
batch_size = 16
dr         = 0.2

num_maps_per_projection_direction = 10
num_sims = 2000
starting_sim_number = 1000

## Load pretrained paramters
model_name = "ViT"
pretrained_dir = f'./saved_models/lightning_logs_csv/version_6'
pretrained_filename = f'{pretrained_dir}/checkpoints/epoch=27-step=84000.ckpt'
filename = f'{pretrained_dir}/hparams.yaml'
model_kwargs = extract_model_params(filename, 'model_kwargs')
lr = extract_model_params(filename, 'lr')
beta1 = extract_model_params(filename, 'beta1')
beta2 = extract_model_params(filename, 'beta2')
wd = extract_model_params(filename, 'wd')

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
            '--smallest_sim_number', f'{starting_sim_number}', '--log_1_plus'
        ]
        #         --precomputed_mean {MEAN} --precomputed_stddev {STD} \
        #         --precomputed_min_vals {MIN_VALS[0]} {MIN_VALS[1]} {MIN_VALS[2]} {MIN_VALS[3]} {MIN_VALS[4]} \
        #         --precomputed_max_vals {MAX_VALS[0]} {MAX_VALS[1]} {MAX_VALS[2]} {MAX_VALS[3]} {MAX_VALS[4]} \

    import subprocess
    result = subprocess.run(command)
    print(result)

# Load the mean, std, min_vals and max_vals into variables
MEAN = np.load(f'{prefix}_dataset_mean.npy')
STD = np.load(f'{prefix}_dataset_std.npy')
MIN_VALS = np.load(f'{prefix}_dataset_min_vals.npy')
MAX_VALS = np.load(f'{prefix}_dataset_max_vals.npy')
MEAN_DENSITIES = np.load(f'{prefix}_dataset_mean_densities.npy')
print(MEAN, STD, MIN_VALS, MAX_VALS)

from torch.utils.data import DataLoader
from utils import CustomImageDataset
from utils import MyRotationTransform

transform = torchvision.transforms.Compose([
    MyRotationTransform(angles=[90, 180, 270]),
])

train_dataset = CustomImageDataset(f'{base_dir}/train', normalized_cosmo_params_path=f'{base_dir}/train/train_normalized_params.csv', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

val_dataset = CustomImageDataset(f'{base_dir}/val', normalized_cosmo_params_path=f'{base_dir}/val/val_normalized_params.csv', transform=None)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

test_dataset = CustomImageDataset(f'{base_dir}/test', normalized_cosmo_params_path=f'{base_dir}/test/test_normalized_params.csv', transform=None)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)


# Updated parameters for the transfer learning come here.
epochs = 20  # Smaller no. of epochs than pretraining.
FREEZE_LAYERS = True  # If True, all layers except Linear in to_patch_embedding are freezed

from pytorch_lightning.loggers import WandbLogger
logger_csv = pl.loggers.CSVLogger(CHECKPOINT_PATH, name="lightning_logs_csv")
WANDB_RUN_NAME = f'ViT_TL_DiffSimsNomasscut-batchsize-{batch_size}_lr-{lr}_epochs-{epochs}_wd-{wd}_dr-{model_kwargs["dropout"]}_freeze-{FREEZE_LAYERS}'
wandb_logger = WandbLogger(name=WANDB_RUN_NAME, project='Cosmo-parameter-inference')
wandb_logger.experiment.config.update({"batch_size": batch_size, "epochs": epochs})

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
        if model_name == "ViT":
            model = ViT_FineTune(pretrained_filename, **kwargs) 
        elif model_name == "CNN":
            model = CNN_FineTune(pretrained_filename, **kwargs) 
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # Automatically loads the model with the saved hyperparameters
    else:
        raise ValueError("Finetuning requires a pretrained model file to be specified by the `PRETRAINED_FILENAME` argument!")
        
    # After loading the pretrained model, finetune it.
    trainer.fit(model, train_loader, val_loader)

    if model_name == "ViT":
        model = ViT_FineTune.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    elif model_name == "CNN":
        model = CNN_FineTune.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_loss"], "val": val_result[0]["test_loss"]}

    return model, result, trainer.checkpoint_callback.best_model_path

## Finetune the model
model, results, FINETUNED_FILENAME = finetune_model(
    pretrained_filename, model_name=model_name, model_kwargs=model_kwargs,
    lr=lr, wd=wd, beta1=beta1, beta2=beta2,
    minimum=MIN_VALS, maximum=MAX_VALS, freeze_layers=FREEZE_LAYERS
)

print("ViT_FineTune results", results)
model.to(device)
print(f'Fine-tuned file name: {FINETUNED_FILENAME}')


#################################################
# Now test the finetuned model.
#################################################

from train_val_test_boilerplate import test

# Below values calculated during data preparation. See above.
minimum = MIN_VALS
maximum = MAX_VALS
params = [0,1,2,3,4]
g = params
h = [5+i for i in g]

params_true, params_NN, errors_NN, filenames = test(model, test_loader, g=g, h=h, device=device, minimum=minimum, maximum=maximum)

from evaluation_analysis import post_test_analysis

post_test_analysis(
    params_true, params_NN, errors_NN, filenames,
    params, num_sims, MEAN, STD, MEAN_DENSITIES, minimum, maximum,
    num_maps_per_projection_direction,
    test_results_filename=f'test_results_transfer_learning_ViT_epoch{epochs}_freeze{FREEZE_LAYERS}.csv',
    smallest_sim_number=0 if SAME_SIMS else starting_sim_number
)

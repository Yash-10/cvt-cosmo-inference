### Step 1: 
Download the snapshots to the supercomputer using Globus. Instructions are https://quijote-simulations.readthedocs.io/en/latest/access.html#globus. You can use globus-cli, which can help downloading the data easily.

We need to access the San Diego cluster since it is the one containing the standard Latin hypercube snapshots. It is located here: https://app.globus.org/file-manager?origin_id=f4863854-3819-11eb-b171-0ee0d5d9299f&origin_path=%2F. In this link, we must go to the path "Snapshots/latin_hypercube". Inside this path, you will see folders from 0-1999. In each of these folders, we only need the snapdir_004 folder (corresponding to redshift = 0) and the Cosmo_params.dat (containing the cosmological parameters) file.

(OPTIONAL: The values of cosmological parameters are also present here: https://github.com/franciscovillaescusa/Quijote-simulations/blob/master/latin_hypercube/latin_hypercube_params.txt - first row corresponds to simulation 0, second corresponds to simulation 1, etc - so one can download that single file instead of downloading Cosmo_params.dat for each simulation. The code might need to be changed if this is opted.)

As a rough estimate, I found that each snapshot might require ~3-4 GB, so in total, for 2000 snapshots, it will require 6-8 TB (it's possible that the calculation is wrong). If this will occupy too much disk space, you can download a subset of the simulations (e.g., the first 1000).

### Step 2: 
Install required packages

```
pip install -r requirements.txt
```

### Step 3: 
Create density fields from the snapshots

(a) `create_density_fields_from_snapshots.py` creates the density fields from 2000 snapshots. You need to run it as `python create_density_fields_from_snapshots.py`

If you are using a lesser no. of simulations than 2000, please change the `NUM_SIMS` variable inside the script accordingly. If you are changing `NUM_SIMS`, you must also change the variable `num_sims` in main_train.py (Step 4) and also the `num_sims` argument to `create_data.py` (see Step 3 below). In any case, could you please keep the same grid size (`grid=256`) since, for this experiment, we want a sufficiently larger grid size compared to 64^3? This script will create a tar zip file as output containing the density fields.

(b) Untar the tar.gz file using tar -xvzf <tar-zip-file>, for example, which will create a folder named my_outputs in the current directory.

### Step 4: 
Create data for the neural network

Run create_data.py as follows:

```
python create_data.py --num_sims 2000 --train_frac 0.8 --test_frac 0.1 --seed 42 --path ./my_outputs --grid_size 256 --num_maps_per_projection_direction 10 --prefix '' --smallest_sim_number 0
```

This will create three folders named `train`, `val`, and `test` in the current directory.

### Step 5: 
Train the ViT

Run `python main_train.py`. All variables including access to the device for training the model (CPU or GPU) are hardcoded in the script itself.

### tep 6: 
Test the ViT

Run `python main_test.py`. This will output the final results in the form of a CSV file and PNG images. Could you share the outputs this script generates?


## Code for transfer learning:

### Steps: 1. 
Download the Halo simulations. The code I have written is slightly inflexible in terms of what Halo simulations to use for transfer learning. 
If you download from 0-1999, SAME_SIMS must be true since you used 0-1999 simulations for the DM density. If you decide to use halo simulations only from 1000-1999, please set SAME_SIMS=False inside main_transfer_learning.py

### Step 2:
Run create_halo_fields_from_snapshots.py (attached).

### Step 3: 
Run main_transfer_learning.py (attached): this will run the transfer learning and testing together. Note: You would need to change some paths, etc, inside the script to successfully run this.




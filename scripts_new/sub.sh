#!/bin/sh

#python create_data.py --num_sims 2000 --train_frac 0.8 --test_frac 0.1 --seed 42 --path ./my_outputs --grid_size 256 --num_maps_per_projection_direction 10 --prefix '' --smallest_sim_number 0 --log_1_plus

python create_data.py --num_sims 2000 --train_frac 0.8 --test_frac 0.1 --seed 40 --path ./my_outputs_128 --grid_size 128 --num_maps_per_projection_direction 10 --prefix 'dens128' --smallest_sim_number 0 --log_1_plus

#python create_data.py --num_sims 2000 --train_frac 0.8 --test_frac 0.1 --seed 40 --path ./my_outputs_halo --grid_size 256 --num_maps_per_projection_direction 10 --prefix 'halos' --smallest_sim_number 0 --log_1_plus --dataset_name 3D_halo_distribution

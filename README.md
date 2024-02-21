# ViT-LSS


## Description of files
- `preprocessing/create_density_fields_grid64.ipynb`: Generates 3D density fields from the first 1000 LH set simulations (out of the total 2000 simulations) from Quijote and saves the tar zip file containing the 3D density fields. No processing of the density fields is performed, i.e., they are stored as output from the mass-assignment procedure. The CIC mass assignment method is used.
- `preprocessing/analyze_density_fields_pspec_64grid.ipynb`: After generating the 3D density fields, 


- These notebooks contain an older version of the CNN approach. So, while it's not helpful to look at the results there, it is kept in this repository because these notebooks contain data exploration plots, which should hold true even if the CNN approach is old.
    - `CNN_DM_0_to_999_&_Halo_0_to_999_no_masscut.ipynb`
    - `CNN_DM_0_to_999_&_Halo_1000_to_1999.ipynb`
    - `CNN_DM_0_to_999_&_Halo_1000_to_1999_1e14_masscut.ipynb`

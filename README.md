# ViT-LSS


## Description of files
- `preprocessing/create_density_fields_grid64.ipynb`: Generates 3D density fields from the first 1000 LH set simulations (out of the total 2000 simulations) from Quijote and saves the tar zip file containing the 3D density fields. No processing of the density fields is performed, i.e., they are stored as output from the mass-assignment procedure. The CIC mass assignment method is used.
- `preprocessing/analyze_density_fields_pspec_64grid.ipynb`: After generating the 3D density fields, 

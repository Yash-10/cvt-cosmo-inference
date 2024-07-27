
import pandas as pd
import numpy as np

fname = "./my_outputs_128/train/train_normalized_params.csv"
data = pd.read_csv(fname)
for idx in range(len(data)):
    value = data.iloc[idx,1]
    print(value)
    data.iloc[idx,1] = value.replace("ViT_inference", "ViT_inference/my_outputs_128")

data.to_csv(fname, index=False)

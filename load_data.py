# Modified function to load data
import os
import pandas as pd
# import numpy as np


def load_data(x_file, y_file):
    data_dir = os.path.abspath("data_fashion/")

    # Load data
    x_df = pd.read_csv(os.path.join(data_dir, x_file)).to_numpy()
    y_df = pd.read_csv(os.path.join(data_dir, y_file))
    
    return x_df, y_df

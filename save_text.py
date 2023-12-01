# Function from Arman that takes numeric readings and ouputs the class names needed for submission.

import os
import numpy as np

def save_text(y_hat, file_name):
    names = ['top', 'trouser','pullover', 'dress','fifth','sandal','sixth','sneaker']
    

    # Load the dataset of interest
    datadir = os.path.abspath('data_fashion')
    x_NF = np.loadtxt(
        os.path.join(datadir, 'x_valid.csv'),
        delimiter=',',
        skiprows=1)
    N = x_NF.shape[0]

    # Create random predictions (just for fun)
    prng = np.random.RandomState(100)
    predictions = []
    for n in range(len(y_hat)):
        result = names[int(y_hat[n])]
        predictions.append(result)

    
    # Save the predictions in the leaderboard format
    np.savetxt(file_name, predictions, delimiter='\n', fmt='%s')
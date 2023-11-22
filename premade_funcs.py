import numpy as np
import pandas as pd
import os
import sklearn.metrics
import matplotlib.pyplot as plt

def load_data(dir_name):
    data_dir = dir_name

    # Load data
    train_x = pd.read_csv(os.path.join(data_dir, "x_train.csv")).to_numpy()
    train_y_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    valid_x = pd.read_csv(os.path.join(data_dir, "x_valid.csv")).to_numpy()
    valid_y_df = pd.read_csv(os.path.join(data_dir, "y_valid.csv"))
    
    test_x = pd.read_csv(os.path.join(data_dir, "x_test.csv")).to_numpy()

    # Print shapes
    for label, arr in [('train', train_x), ('valid', valid_x), ('test', test_x)]:
        print("Contents of %s_x.csv: arr of shape %s" % (
            label, str(arr.shape)))
        
    return test_x, train_x, train_y_df, valid_x, valid_y_df


#save predictions
def format_pred(dir_name, predictions):
    names = ['dress', 'pullover', 'top', 'trouser', 'sandal', 'sneaker']

    # Load the dataset of interest
    datadir = dir_name
    x_NF = np.loadtxt(
        os.path.join(datadir, 'x_valid.csv'),
        delimiter=',',
        skiprows=1)
    N = x_NF.shape[0]

    predictions = predictions
    for n in range(N):
        pred_name = prng.choice(names, size=1)
        predictions.append(rand_name)

    # Save the predictions in the leaderboard format
    np.savetxt('yhat_valid.txt', predictions, delimiter='\n', fmt='%s')


def calc_accuracy(dir_name, y_df, ytrue_N, ):
    datadir = dir_name

    # Load true labels
    y_df = pd.read_csv(os.path.join(datadir, 'y_valid.csv'))
    ytrue_N = y_df['class_name'].values

    format_pred(dir_name)
    # Load predictions
    try:
        yhat_N = np.loadtxt('yhat_valid.txt', dtype=str)
    except IOError:
        raise ValueError("Did you run save_rand_predictions.py first??")

    assert ytrue_N.shape == yhat_N.shape

    print("Loaded true and predicted labels")
    disp_df = pd.DataFrame(np.hstack([yhat_N[:,np.newaxis], ytrue_N[:,np.newaxis]]),
        columns=['yhat', 'ytrue'])
    print(disp_df)
    
    bal_acc = sklearn.metrics.balanced_accuracy_score(ytrue_N, yhat_N)
    print("")
    print("Balanced Accuracy: %.3f" % bal_acc)
    print("remember, balanced accuracy for a random guess should be (in expectation) 1/C = 1/6 = %.3f" % (1/6.))
    return  bal_acc
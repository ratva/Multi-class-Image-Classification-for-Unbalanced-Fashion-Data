import numpy as np
import pandas as pd
import os

def load_data(x_data, y_data):
    data_dir = 'data_reviews'
    x_df = pd.read_csv(os.path.join(data_dir, x_data))
    y_df = pd.read_csv(os.path.join(data_dir, y_data))
    N, n_cols = x_df.shape
    # print("Shape of x_df: (%d, %d)" % (N,n_cols))
    # print("Shape of y_df: %s" % str(y_df.shape))
    # print(type(x_df))
    # print(type(y_df))
    # Print out the first five rows and last five rows
    website_list = x_df['website_name'].values.tolist()
    text_list = x_df['text'].values.tolist()
    rating_list = y_df['is_positive_sentiment'].values.tolist()
    # print(text_list)
    return website_list, text_list, rating_list

# if __name__ == '__main__':
#     data_dir = 'data_reviews'
#     x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
#     y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

#     N, n_cols = x_train_df.shape
#     print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
#     print("Shape of y_train_df: %s" % str(y_train_df.shape))

#     # Print out the first five rows and last five rows
#     tr_text_list = x_train_df['text'].values.tolist()
#     rows = np.arange(0, 5)
#     for row_id in rows:
#         text = tr_text_list[row_id]
#         print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

#     print("...")
#     rows = np.arange(N - 5, N)
#     for row_id in rows:
#         text = tr_text_list[row_id]
#         print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

import numpy as np
import pandas as pd


# Load data
data_loc = 'IRIS.csv'
iris_df = pd.read_csv(data_loc)

X_df = iris_df.drop('species', axis=1)
y_df = iris_df['species']

# Train & Test Spliting
train_ratio = 0.7
train_rows = int(X_df.shape[0] * train_ratio)

X_train = X_df.iloc[:train_rows]
X_test = X_df.iloc[train_rows:]

y_train = y_df.iloc[:train_rows]
y_test = y_df.iloc[train_rows:]




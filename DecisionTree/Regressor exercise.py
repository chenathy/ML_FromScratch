# Load Modules
import numpy as np
import pandas as pd

from DecisionTree.Regressor import DecisionTree


# Data Prep
#### Read Data
data_loc = 'Airfoil Self Noise.csv'
airfoil = pd.read_csv(data_loc)
airfoil = airfoil.sample(frac=1, random_state=1234)
X = airfoil.drop('SSPL', axis=1)
y = airfoil['SSPL']


#### Train Test Split
train_test_ratio = 0.7
train_rows = int(airfoil.shape[0] * train_test_ratio)
X_train, y_train = X.iloc[:train_rows, :], y[:train_rows]
X_test, y_test = X.iloc[train_rows:, :], y[train_rows:]



# Train Model
regressor = DecisionTree(max_depth=3, min_sample_split=3)
regressor.fit(X_train, y_train)


# Visual Model
regressor.print_tree(regressor.root)


# Predict
y_pred = regressor.predict(X_test)


# Evaluate Model Performance
mse = np.mean(np.square(y_pred - y_test))
print(f'Mean Squared Root Error: {np.sqrt(mse):.5f}')
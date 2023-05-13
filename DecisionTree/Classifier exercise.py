# Load Modules
import numpy as np
import pandas as pd

from DecisionTree.Classifier import DecisionTree


# Data Prep
#### Read Data
data_loc = 'IRIS.csv'
iris = pd.read_csv(data_loc)
iris = iris.sample(frac=1, random_state=1234)
X = iris.drop('species', axis=1)
y = iris['species']

#### Train Test Split
train_test_ratio = 0.7
train_rows = int(iris.shape[0] * train_test_ratio)
X_train, y_train = X.iloc[:train_rows, :], y[:train_rows]
X_test, y_test = X.iloc[train_rows:, :], y[train_rows:]


# Train Model
classifier = DecisionTree(max_depth=3, min_sample_split=3, criteria='entropy')
classifier.fit(X_train, y_train)


# Visual Tree
classifier.print_tree()


# Predict
y_pred = classifier.predict(X_test)


# Performace
accuracy = np.sum(y_test == y_pred) / len(y_test)
print(f'Accuracy of model is {accuracy * 100:.2f}%')





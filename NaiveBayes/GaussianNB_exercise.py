# Load Modules
import numpy as np
import pandas as pd

from NaiveBayes.GaussianNB import GaussianNB


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
clf = GaussianNB()
clf.fit(X_train, y_train)


# Predict
y_pred = clf.predict(X_test)


# Performance
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Accuracy of Entire Forest (All Trees) : {accuracy * 100:.2f}%')

import pandas as pd
import numpy as np
from SVM.svm import LinearSVM


# Load data
data_loc = 'IRIS.csv'
iris_df = pd.read_csv(data_loc)


# Only selecting the 2 species data to see performance
outcomes_first2 = iris_df.species.unique()[:2]
# >>> array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

df = iris_df[
    iris_df.species.isin(
        outcomes_first2
    )
]
df = df.sample(frac=1, random_state=42)
X = df.drop('species', axis=1)
y = df['species']

# Train & Test Splitting
# Set the seed for reproducibility
np.random.seed(42)

# Generate a random mask for the split
mask = np.random.rand(len(df)) < 0.8

X_train = X[mask]
X_test = X[~mask]

y_train = y[mask]
y_test = y[~mask]


# Train Model
svm_linear = LinearSVM()
svm_linear.fit(X_train, y_train)


# Test Model Performance
y_pred = svm_linear.predict(X_test)

accuracy = np.mean(y_pred == y_test)

print(f"Accuracy Rate: {accuracy * 100:.2f}%")
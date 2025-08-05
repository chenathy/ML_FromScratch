# Load Modules
import pandas as pd
from Perceptron.perceptron import *


# Data Prep
#### Read Data
data_loc = 'Heart Disease.csv'
heart_df = pd.read_csv(data_loc)
X = heart_df[['thalach', 'oldpeak']]
y = heart_df['target']


### Train & Test Spliting
train_ratio = 0.72
train_rows = int(X.shape[0] * train_ratio)

np.random.seed(42)
X_indexes = list(X.index)

train_indexes = np.random.choice(X_indexes, size=train_rows, replace=False)
test_indexes = [i for i in X_indexes if i not in train_indexes]

X_train = X.iloc[train_indexes]
X_test = X.iloc[test_indexes]

y_train = y.iloc[train_indexes]
y_test = y.iloc[test_indexes]


### Normalize the training and testing data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()


## Training Data
lr = 0.01
epochs = 20
weights_trained = train_data(
    train_x=X_train,
    y=y_train,
    learning_rate=lr,
    epoches=epochs
)


### Evaluation
y_pred = predict_data(X_test, weights_trained)
accuracy = np.mean(y_pred == np.array(y_test))
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
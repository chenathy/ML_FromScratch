import pandas as pd
import numpy as np
from AdaBoost.ada_boost import Ada_Boost


# Data Prep
#### Read Data
breast_cancer_df = pd.read_csv('Breast Cancer (numerical).csv')
X = breast_cancer_df.drop(['diagnosis', 'id'], axis=1)
y = breast_cancer_df['diagnosis'].map({'B': 1, 'M': -1})

#### Train Test Split
train_test_ratio = 0.7
train_rows = int(X.shape[0] * train_test_ratio)
X_train, y_train = X.iloc[:train_rows, :], y[:train_rows]
X_test, y_test = X.iloc[train_rows:, :], y[train_rows:]


# Train Model
ada_model = Ada_Boost(n_rounds=5)
ada_model.fit(X_train, y_train)


# Predict
y_pred = ada_model.predict(X_test)


# Evaluate Model Performance
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Accuracy of Ada Boost, with {len(ada_model.clfs)} Decision Stump : {accuracy * 100:.2f}%')


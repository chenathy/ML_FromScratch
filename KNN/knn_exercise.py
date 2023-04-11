

# 1. Load the dataset:
# Load the training dataset into memory.
# This dataset should contain labeled examples of inputs and outputs.

import pandas as pd
import numpy as np

data_loc = './KNN/KNNAlgorithmDataset.csv'
df = pd.read_csv(data_loc)
df.drop('Unnamed: 32', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)


# 2. Choose the value of k:
# Choose a value for k, which represents the number of nearest neighbors to consider when making a prediction.
# This value should be an odd number to avoid ties in binary classification problems.
k = 3


# 3. Choose a distance metric:
# Choose a distance metric, such as Euclidean distance or Manhattan distance,
# to measure the distance between the input data and the training data.
dist = 'Euclidean'


# 4. Preprocess the data:
# Preprocess the input data to ensure that it is in the same format as the training data.
# This can involve steps such as scaling or normalizing the features, handling missing values, or encoding categorical variables.
train_ratio = 0.7
train_rows = int(df.shape[0] * train_ratio)

# Features and Label
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']  # .replace({'M': 0, 'B': 1})

# Train Test Split
X_train = X.iloc[:train_rows]
y_train = y[:train_rows]

X_test = X.iloc[train_rows:]
y_test = y[train_rows:]


# Normalize both train and input data
def scale_features(train_df, test_df):

    for col in train_df.columns:

        print(f'normalizing feature {col} ...')
        mean_train = np.mean(train_df[col])
        std_train = np.std(train_df[col])

        # standardize training data
        train_df[col] = (train_df[col] - mean_train) / std_train

        # standardize testing data
        test_df[col] = (test_df[col] - mean_train) / std_train

    print('Finished Normalizing both Train and Test data')
    return train_df, test_df

X_train, X_test = scale_features(X_train, X_test)



# 5. Calculate distances:
# Calculate the distance between the input data and each training data point using the chosen distance metric.
# should return m x n matrix (m: test, n: train)

def dist_calculation(train_data, test_data, dist_type = 'Euclidean'):

    # matrix the train_data
    if not isinstance(train_data, np.ndarray):
        train_data = train_data.to_numpy()

    if not isinstance(test_data, np.ndarray):
        test_data = test_data.to_numpy()

    rows_train, cols = train_data.shape
    print(f'Training Data has {rows_train} rows')

    if test_data.ndim == 1:
        rows_test = 1
        print(f'Testing Data is single row data with {test_data.shape[0]} features array')
    else:
        rows_test = test_data.shape[0]
        print(f'Testing Data has {rows_test} rows')




    if dist_type == 'Euclidean':

        if rows_test == 1:
            result_matrix = np.sqrt(
                    np.sum([
                        np.square(
                            train_data[:, col] - test_data[col]
                        )
                        for col in range(cols)
                    ],
                        axis= 0
                    )
            )
        else:
            result_matrix = [
                [
                    np.sqrt(
                        np.sum([
                            np.square(
                                train_data[row_train, col] - test_data[row_test, col]
                            )
                            for col in range(cols)
                        ])
                    )
                    for row_train in range(rows_train)
                ]
                for row_test in range(rows_test)
            ]

    elif dist_type == 'Manhattan':

        if rows_test == 1:
            result_matrix = np.sum([
                    np.abs(train_data[:, col] - test_data[col])
                    for col in range(cols)
                ],
                    axis= 0
            )

        else:
            result_matrix = [
                [
                    np.sum([
                            np.abs(train_data[row_train, col] - test_data[row_test, col])
                            for col in range(cols)
                        ])
                    for row_train in range(rows_train)
                ]
                for row_test in range(rows_test)
            ]

    result_matrix = np.array(result_matrix)

    if result_matrix.ndim == 1:
        print((f'`Distance Matrix` returned single row of {result_matrix.shape[0]} elements array'))
    else:
        print(f'`Distance Matrix` returned {result_matrix.shape[0]} rows and {result_matrix.shape[1]} columns')
    return result_matrix


distance_matrix = dist_calculation(X_train, X_test, dist_type= dist)


# 6. Choose the k-nearest neighbors:
# Select the k training data points
# that are closest to the input data based on the distance metric.


def nearest_indexes(k, distance_matrix):

    if distance_matrix.ndim == 1:
        indexes_k = [list(np.argsort(distance_matrix)[:k])]

    else:
        length = distance_matrix.shape[0]

        indexes_k = [
            list(
                np.argsort(
                    distance_matrix[i]
                )[:k]
            )
            for i in range(length)
        ]

    return indexes_k

near_indexes = nearest_indexes(k=k, distance_matrix=distance_matrix)

# 7. Make a prediction:
# For classification problems,
# count the number of examples in each class among the k-nearest neighbors, and predict the class with the highest count.
#
# For regression problems,
# take the average of the output values among the k-nearest neighbors and use this as the predicted output value.


def predict_from_indexes(near_indexes):

    length = len(near_indexes)

    top_k_pred = [
        list(y_train[near_indexes[i]])
        for i in range(length)
    ]

    pred = [
        max(top_k_pred[i], key=top_k_pred[i].count)
        for i in range(length)
    ]

    return pred

y_pred = predict_from_indexes(near_indexes=near_indexes)

# 8. Evaluate the model:
# Evaluate the performance of the model using a separate validation dataset or through cross-validation.

# compare 2 lists:
# y_pred, y_test

def accuracy(list1, list2):

    assert (len(list1) == len(list2))

    # Calculate the number of elements that are consistent between the two lists
    correct_num = sum([1 for i, j in zip(list1, list2) if i == j])

    # Calculate the accuracy as a percentage
    accuracy = 100 * correct_num / len(list2)

    print("Number of correctly predicted:", correct_num)
    print("Accuracy: {:.2f}%".format(accuracy))

    # return num_consistent, accuracy


accuracy(y_pred, y_test)




# 9. Tune the model:
# Adjust the hyperparameters of the KNN algorithm,
# such as the value of k or the choice of distance metric,
# to improve its performance on the validation dataset.

# -------------------------
k = 5
dist = 'Euclidean'

distance_matrix = dist_calculation(X_train, X_test, dist_type= dist)

near_indexes = nearest_indexes(k=k, distance_matrix=distance_matrix)

y_pred = predict_from_indexes(near_indexes=near_indexes)

accuracy(y_pred, y_test)

# ------------------------
k = 3
dist = 'Manhattan'

distance_matrix = dist_calculation(X_train, X_test, dist_type= dist)

near_indexes = nearest_indexes(k=k, distance_matrix=distance_matrix)

y_pred = predict_from_indexes(near_indexes=near_indexes)

accuracy(y_pred, y_test)



# 10. Make predictions: Once the model has been trained and tuned,
# it can be used to make predictions on new input data.

def predict_Obs(input_data, train_data, k, dist_type):

    distance_matrix = dist_calculation(train_data, input_data, dist_type=dist_type)

    near_indexes = nearest_indexes(k=k, distance_matrix=distance_matrix)

    y_pred = predict_from_indexes(near_indexes=near_indexes)

    return y_pred



predict_Obs(
    input_data=X_test.iloc[20],
    train_data=X_train,
    k=k,
    dist_type=dist
)





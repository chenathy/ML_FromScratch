
#### Steps

# 1. .

# 2.

# 3.

# 4.

# 5. Find nearest neighbors: Select the K-nearest neighbors based on the calculated distances.

# 6. Classify/regress: For classification problems, determine the majority class among the K-nearest neighbors and assign that class to the test data point. For regression problems, determine the average of the K-nearest neighbors and assign that value to the test data point.

# 7. Evaluate the model: Once the model has been trained and tested, evaluate its performance using metrics such as accuracy, precision, recall, F1 score, or mean squared error (MSE).

# 8. Tune hyperparameters: Adjust the hyperparameters, such as K or the distance metric, to optimize the model's performance.

# 9. Make predictions: Use the trained model to make predictions on new, unseen data points.



# 1. Load the dataset: Load the training dataset into memory. This dataset should contain labeled examples of inputs and outputs.

# 2. Choose the value of k: Choose a value for k, which represents the number of nearest neighbors to consider when making a prediction. This value should be an odd number to avoid ties in binary classification problems.

# 3. Choose a distance metric: Choose a distance metric, such as Euclidean distance or Manhattan distance, to measure the distance between the input data and the training data.

# 4. Preprocess the data: Preprocess the input data to ensure that it is in the same format as the training data.

# 5. Calculate distances: Calculate the distance between the input data and each training data point using the chosen distance metric.

# 6. Choose the k-nearest neighbors: Select the k training data points that are closest to the input data based on the distance metric.

# 7. Make a prediction: For classification problems, count the number of examples in each class among the k-nearest neighbors, and predict the class with the highest count. For regression problems, take the average of the output values among the k-nearest neighbors and use this as the predicted output value.

# 8. Evaluate the model: Evaluate the performance of the model using a separate validation dataset or through cross-validation.

# 9. Tune the model: Adjust the hyperparameters of the KNN algorithm, such as the value of k or the choice of distance metric, to improve its performance on the validation dataset.

# 10. Make predictions: Once the model has been trained and tuned, it can be used to make predictions on new input data.


######## Step 1 ###########
# Load the dataset:
## Load the training dataset into memory.
## This dataset should contain labeled examples of inputs and outputs.
import pandas as pd
import numpy as np

data_loc = './KNNAlgorithmDataset.csv'
df = pd.read_csv(data_loc)
df.drop('Unnamed: 32', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)

df_columns = list(df.columns)

# category variable
quality_vars = ['diagnosis']
quantity_vars = [i for i in df_columns if i not in quality_vars]

# preprocess categorical variable
df['diagnosis'].replace({'M': 0, 'B': 1}, inplace=True)

# preprocess data normalization
def normalizeData(data):

    # get mean and std
    mean = np.mean(data)
    std = np.std(data)

    # normalize data
    data_normalized = (data - mean)/std

    return data_normalized

for i in quantity_vars:
    print(f'Normalizing column {i} ...')
    df[i] = normalizeData(df[i])




####### Step 2 ############
# Split the data: Divide the dataset into training and test sets.
train_ratio = 0.7
train_rows = int(df.shape[0] * train_ratio)

df_train = df.iloc[:train_rows]
df_test = df.iloc[train_rows:]



####### Step 3 #############
# Define K: Choose the number of neighbors (K) to consider for classification or regression.
K = 3


####### Step 4 #############
# Calculate distance:
# Calculate the distance between the test data point and all other points in the training set
# using a distance metric (such as Euclidean or Manhattan distance).

def dist_calculation(data, distance_type = 'Euclidean'):

    # calculate the Euclidian distance of one point to others
    rows, cols = data.shape

    for row in range(rows-1):
        current_df = data.iloc[row]
        rest_df = data.iloc[row+1:, ].to_numpy()

        ## substract current value to each column
        for col in range(cols-1):
            rest_df[col, :] = rest_df[col, :] - current_df[col]

        if distance_type == 'Euclidean':
            ## square each element
            rest_df = np.square(rest_df)
            sum = np.sum(rest_df)

        else:
            # distance_type == 'Manhattan':
            ## abs value each element
            rest_df = np.abs(rest_df)
            sum = np.sum(rest_df)

    sum = sum / rows
    return sum










matrix_test = np.matrix(
    [[1, 4, 7],
     [2, 5, 8],
     [3, 6, 9]]
)
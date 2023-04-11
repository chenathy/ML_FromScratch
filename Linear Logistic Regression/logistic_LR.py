import numpy as np
import pandas as pd

from StatsFuncs.statistics import Statistics


############################################
####   Gather and preprocess the data:  ####
############################################
# Collect the dataset and preprocess it by cleaning, normalizing, and transforming it to make it suitable for logistic regression.
# This involves removing missing values, handling categorical variables, and scaling the numerical variables, etc.

data_loc = './Heart Disease.csv'
heart_disease = pd.read_csv(data_loc)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']


############################################
####          Split the data:          #####
############################################
# Divide the data into training and test sets.
# The training set is used to train the model, while the test set is used to evaluate the performance of the model.
train_test_ratio = 0.7
train_rows = int(heart_disease.shape[0] * train_test_ratio)

X_train, X_test = X.iloc[:train_rows, ], X.iloc[train_rows:, ]
y_train, y_test = y[:train_rows], y[train_rows:]


############################################
##  Define the logistic regression model  ##
############################################
# In this step, you define the logistic regression model.
# The model is a function that takes in the
# input features and outputs the probability of the target variable being in one of the possible classes.


# One Hot Encoded
columns_to_encoded = [
    'cp',
    'restecg',
    'slope',
    'ca',
    'thal'
]


for col in columns_to_encoded:
    print(f'Encoding column {col}...')
    X_train = Statistics.merge_encoded_df(X_train, col)
    X_test = Statistics.merge_encoded_df(X_test, col)



# Assign a bias column to features
X_train = Statistics.assign_bias(X_train)
X_test = Statistics.assign_bias(X_test)



# Standardize the data
colummns_to_standard = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak'
]

X_train[colummns_to_standard], X_test[colummns_to_standard] = \
    Statistics.scale_features(X_train[colummns_to_standard], X_test[colummns_to_standard])



############################################
####           Train the model:         ####
############################################
# Use the training set to fit the model by adjusting the model parameters to minimize the loss function.
# The loss function measures the difference between the predicted and actual values of the target variable.


def train_model(X, y, lr, max_iterations, converge_threshold=1e-4):


    # Initialize the weights to zero or small random values.
    weights = np.zeros(X.shape[1])

    loss = 1

    for i in range(max_iterations):

        # Calculate the dot product of X and weights
        z = X @ weights

        # Pass the dot product through sigmoid function to get the predicted probability
        y_pred = Statistics.sigmoid(z)

        # Calculate the error between predicted probability and true label
        error = y_pred - y

        # Calculate the gradient of weights
        gradient = X.T @ error / len(y)

        # Update the weights
        weights -= lr * gradient

        # Update Loss
        loss_updated = (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()
        # Converge
        if np.abs(loss_updated - loss) <= converge_threshold:
            print(f'Model Converged at iteration {i}...while its Loss: {loss_updated:.4f}')
            break
        else:
            loss = loss_updated


        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {loss:.4f}')

    return weights




# Set the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 20000
converge_threshold = 1e-5


weights = train_model(
    X_train,
    y_train,
    lr=learning_rate,
    max_iterations=num_iterations,
    converge_threshold=converge_threshold
)





############################################
####         Evaluate the model:       #####
############################################
# Use the test set to evaluate the performance of the model.
# The performance can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

# Calculate the predicted probabilities for the testing set
y_pred_test = Statistics.sigmoid(X_test @ weights)
# Convert the probabilities to binary class labels
y_pred_test_binary = np.where(y_pred_test > 0.5, 1, 0)

# Calculate Confusion Matrix
confusion_matrix = Statistics.confusion_matrix(y_test, y_pred_test_binary)
print(f'Confusion Matrix is: \n {confusion_matrix}')

# Calculate the accuracy of the predictions
accuracy_matrix = Statistics.accuracy_metrics(y_test, y_pred_test_binary)
print(f'Accuracy Matrix is: \n {accuracy_matrix}')

# accuracy = (y_pred_test_binary == y_test).mean()
# print(f'Accuracy from testing dataset is {accuracy * 100:.4f}%')





############################################
####           Make predictions:        ####
############################################
# Use the trained model to make predictions on new data.
new_data = X_test.iloc[-1:, ]
new_data_pred = Statistics.sigmoid(new_data @ weights)







############################################
####    Iterate and improve the model:  ####
############################################
# If the performance of the model is not satisfactory, you can iterate and improve the model
# by adjusting the model parameters, changing the preprocessing steps,
# or changing the model architecture until you achieve the desired performance.


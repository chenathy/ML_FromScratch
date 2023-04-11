import pandas as pd
import numpy as np


# Load data  ################################
data_loc = 'Boston Housing.csv'
housing_df = pd.read_csv(data_loc)


X = housing_df.drop('MEDV', axis=1)
y = housing_df['MEDV']


# Train & Test Spliting ######################
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



## Step 0
#########################################
######  Merge Bias into Weights  ########
#########################################
# Add a column of ones to the training data for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Normalize the training and testing data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()


# Step 1
# Initialize the weight vector with random values.
def init_parameters(features_num):

    w = np.random.rand(features_num)

    return w

w0 = init_parameters(X_train.shape[1])


# Step 2
# Calculate the predicted values of the output variable using the current model parameters
# (i.e., the bias term and weight vector) for all training examples.
def predict_w_params(X, weights):

    assert(X.shape[1] == len(weights))

    predicted = X @ weights

    return predicted

y_hat = predict_w_params(X_train, w0)


# Step 3
# Calculate the error between the predicted values and the true output variable values for all training examples.
def error_evaluate(X, y, weights):

    y_hat = predict_w_params(X, weights)
    error = y_hat - y

    print(f'SSE: {np.sum(error ** 2)}')
    print(f'MSE: {np.mean(error ** 2)}')

    return error

error0 = error_evaluate(X_train, y_train, w0)



# Step 4
# Calculate the gradient of the cost function
# with respect to the bias term and weight vector using the error values and the input features.
def gradient_descent(X, error):

    m = X.shape[0]

    dw = (X.T @ error) / m

    return dw

dw = gradient_descent(X_train, error0)



# Step 5
# Update the bias term and weight vector by subtracting the gradient multiplied by the learning rate alpha.
learning_rate = 1e-2
w1 = w0 - dw * learning_rate



# Step 6
# Repeat steps 2-5 for a fixed number of iterations or until the model converges
# (i.e., the cost function no longer decreases significantly).
def train_params(X, y, weights, learning_rate, epochs):

    converge_threshold = 1e-1
    sse = 0

    for i in range(epochs):

        print(f'epoch {i + 1}', '=' * 40)
        error = error_evaluate(X, y, weights)
        dw = gradient_descent(X, error)

        # Update parameters
        weights -= learning_rate * dw

        # Calculate updated SSE
        print('Updating----------')
        error_update = error_evaluate(X, y, weights)
        sse_updated = np.sum(error_update ** 2)
        print(f' SSE delta = {sse_updated - sse} ')

        # Converge Test
        if abs(sse_updated - sse) < converge_threshold:
            print(f'Converged at epoch {i + 1}')
            break

        sse = sse_updated

    return weights


max_epochs = 100000
tuned_w = train_params(
    X=X_train,
    y=y_train,
    weights=np.zeros(X_train.shape[1]),
    learning_rate=learning_rate,
    epochs=max_epochs
)




# Step 7
# Use the trained model to make predictions on new data by calculating the dot product between the input feature vector and the weight vector,
# adding the bias term, and returning the result.

def r_squared(y_true, y_pred):

    # Calculate the total sum of squares (TSS)
    tss = np.sum((y_true - np.mean(y_true))**2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred)**2)

    # Calculate the R-squared value
    r2 = 1 - (rss / tss)

    return r2


r_square = r_squared(
    y_test,
    predict_w_params(X_test, tuned_w)
)


print(f"R-squared on test set: {r_square}")


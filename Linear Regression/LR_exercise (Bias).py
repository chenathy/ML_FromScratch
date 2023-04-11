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




# Step 0
# Scale data
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



# Step 1
# Initialize the bias term and weight vector with random values.
def init_parameters(features_num):

    b = np.random.rand()
    w = np.random.rand(features_num)

    return b, w

b0, w0 = init_parameters(X_train.shape[1])

# Step 2
# Calculate the predicted values of the output variable using the current model parameters
# (i.e., the bias term and weight vector) for all training examples.
def predict_w_params(X, bias, weights):

    assert(X.shape[1] == len(weights))

    predicted = bias + (X @ weights)

    return predicted

y_hat = predict_w_params(X_train, b0, w0)


# Step 3
# Calculate the error between the predicted values and the true output variable values for all training examples.
def error_evaluate(X, y, bias, weights):

    y_hat = predict_w_params(X, bias, weights)
    error = y_hat - y

    print(f'SSE: {np.sum(error ** 2)}')
    print(f'MSE: {np.mean(error ** 2)}')

    return error

error = error_evaluate(X_train, y_train, b0, w0)


# Step 4
# Calculate the gradient of the cost function
# with respect to the bias term and weight vector using the error values and the input features.
def gradient_descent(dataframe, error):

    m = dataframe.shape[0]

    dw = (dataframe.T @ error)  / m
    db = np.sum(error) / m

    return db, dw

db, dw = gradient_descent(X_train, error)


# Step 5
# Update the bias term and weight vector by subtracting the gradient multiplied by the learning rate alpha.
learning_rate = 1e-2
b1 = b0 - db * learning_rate
w1 = w0 - dw * learning_rate


# step 6
# Repeat steps 2-5 for a fixed number of iterations or until the model converges
# (i.e., the cost function no longer decreases significantly).

def update_param(X, y, bias, weights, learning_rate):

    converge_threshold = 1e-1
    max_epochs = 100000
    sse = 0

    for i in range(max_epochs):

        print(f'epoch {i + 1}', '=' * 40)
        error = error_evaluate(X, y, bias, weights)
        db, dw = gradient_descent(X, error)

        # Update parameters
        bias -= learning_rate * db
        weights -= learning_rate * dw

        # Calculate updated SSE
        print('Updating----------')
        error_update = error_evaluate(X, y, bias, weights)
        sse_updated = np.sum(error_update ** 2)
        print(f' SSE delta = {abs(sse_updated - sse)} ')

        # Converge Test
        if abs(sse_updated - sse) < converge_threshold:
            print(f'Converged at epoch {i+1}')
            break

        sse = sse_updated

    return bias, weights


tuned_b, tuned_w = update_param(
    X_train,
    y_train,
    b0,
    w0,
    learning_rate
)

# Step 7
# Use the trained model to make predictions on new data
# by calculating the dot product between the input feature vector and the weight vector,
# adding the bias term, and returning the result.
## Test Model Performance
def r_squared(y_true, y_pred):
    # Calculate the total sum of squares (TSS)
    tss = np.sum((y_true - np.mean(y_true))**2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred)**2)

    # Calculate the R-squared value
    r2 = 1 - (rss / tss)

    return r2

r_square = r_squared(y_test, predict_w_params(X_test, tuned_b, tuned_w))

print(f"R-squared on test set: {r_square}")



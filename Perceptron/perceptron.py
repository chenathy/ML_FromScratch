import pandas as pd
import numpy as np


def add_bias_to_input(train_x):
    """
    Adding a bias column to matrix train X
    :param [x]:
    :return: [1, x]
    """

    # Convert matrix into numpy n-dim arrays
    if isinstance(train_x, np.ndarray):
        train_x = np.array(train_x)

    if train_x.ndim > 1:
        # Multiple row matrix
        return np.hstack((np.ones((train_x.shape[0], 1)), train_x))
    else:
        # Single row array
        return np.insert(train_x, 0, 1)


def activation_func(input: float, func_name='relu') -> float:

    if func_name == 'sigmoid':
        return (np.exp(input * -1) + 1)**(-1)

    elif func_name == 'relu':
        return 0 if input < 0 else 1

    else:
        print('Activation function name is not provided')


def train_data(train_x, y, learning_rate=0.05, epoches=10):
    """
    Step 0:
        X (training data): m x n
            m rows, n features

        y (label data): 1 x m
            single row of binary data (i.e. 0 or 1 only)

        learning rate: 0.05
        epoches: # of model training through entire data

    Step 1: Initializing the weights with 0s: 1 x n
        single row: n weights (n + 1) with bias added

    Step 2: Calculate the initial Z
        y_pred = weight * train_x

    Step 3: Apply Activation Func to Z

    Step 4: Determine error
        difference between y and y_pred
        single row: 1 x m

    Step 5: Update the weights
        With error obtained from previous step, update them with learning rate
        delta = error x InputX

    Step 6: Loop entire data until reached epoches or error is 0

    Step 7: Return trained weights
         Bias is included in weights, 1st element

    :param train_x, y, learning_rate, epoches
    :return: updated weights
    """

    # Append Bias column to train data
    train_x = add_bias_to_input(train_x)

    # Step 1
    weights = np.zeros((1, train_x.shape[1]))

    for epoch in range(epoches):
        # Step 2
        z_array = np.dot(weights, train_x.transpose())

        # Step 3
        y_pred = np.array([activation_func(z) for z in z_array[0]])

        # Step 4
        error = np.array(y - y_pred)

        # Step 5
        weights += np.dot(error, train_x) * learning_rate

    # Step 7
    return weights



def predict_data(val_x, trained_weights):
    """
    Validation X: m x n
        m rows, n features

    Trained Weights:
        single row of weights: 1 x n
        n + 1 after adding bias

    :param val_x:
    :param trained_weights:
    :return:
        single row of binary prediction
        1 x m
    """

    # Append bias column to input
    val_x = add_bias_to_input(val_x)

    z_array = np.dot(trained_weights, val_x.transpose())
    return np.where(z_array > 0, 1, 0)




# Unit test
## Adding Bias to Input
sample_input = np.random.rand(20, 5)
sample_input_int = np.random.randint(1, 10, (20, 5))
sample_single_row = np.random.rand(1, 5)

## Activation Func
test_z = 2
activation_func(2)

## Train data
y = np.random.randint(0, 2, (1, 20))
seudo_weights = train_data(
    sample_input,
    y
)


## Predict Data
predict_data(sample_input, seudo_weights)

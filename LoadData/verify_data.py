import numpy as np


def loaded_data_verification(data: np.array):
    """
    Splitting data into X and y

    Verify X and y have the right shape
    """

    n_samples, n_features = data.shape
    n_features -= 1

    X = data[:, 0:n_features]
    y = data[:, n_features]

    print(f'X.shape: {X.shape}')
    print(f'y.shape: {y.shape}')
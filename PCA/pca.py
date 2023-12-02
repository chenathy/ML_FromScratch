import numpy as np
import pandas as pd

class PCA:

    def __init__(self, dataframe, n_components: int):

        self.dataframe = dataframe
        self.obs, self.features = dataframe.shape
        if n_components <= self.features:
            self.n_components = n_components
        else:
            raise ValueError(f'principle components ({n_components}) cannot be greater than features numbers {self.obs}')

    def _substract_mean(self):
        """
        :return: produces a data set whose mean is zero.
        Subtracting the mean makes variance and covariance calculation easier by simplifying their equations.
        The variance and co-variance values are not affected by the mean value.
        """
        print('Centering Data....')
        means_df = self.dataframe.mean(axis=0)
        self.zero_mean_df = self.dataframe - means_df

    def _covariance_matrix(self):
        print('Calculating Covariance of features...')
        self.cov = self.zero_mean_df.T @ self.zero_mean_df / (self.obs - 1)

    def _eigenvalues_eigenvectors(self):
        print('Calculating Eigenvalues and corresponding Eigenvectors...')
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)

        # rank (sort) by eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]


    def _select_principle_components(self):
        print('Selecting PC based on components selection...')
        self.pca = self.eigenvectors[:, : self.n_components]

    def transform_data(self):

        self._substract_mean()
        self._covariance_matrix()
        self._eigenvalues_eigenvectors()
        self._select_principle_components()

        self.pca_data = self.zero_mean_df @ self.pca
        self.pca_data.rename(
            columns={i: f'PC{str(i+1)}' for i in range(self.n_components)},
            inplace=True
        )
        print('PCA transformed data is returned...')
        return self.pca_data


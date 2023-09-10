import numpy as np

class LDA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.linear_discriminates = None


    def fit(self, X, y):

        class_data = [X[y == i] for i in np.unique(y)]

        if self.n_components is None:
            self.n_components = len(np.unique(y)) - 1

        # Compute statistics
        ## overall
        overall_mean = np.mean(X, axis=0)

        ## each class (matrix size of:  classes x features )
        class_means = [np.mean(data) for data in class_data]


        # Compute within class scatter ( features x features )
        # Sw = ∑ ∑ (X - µi) · (X - µi)'
        # Verify (∑ cov(Xi) * Ni): sum([(data - mean).T @ (data - mean) for (data, mean) in zip(class_data, class_means)])
        within_class_scatter = sum([np.cov(data, rowvar=False, ddof=0) * data.shape[0] for data in class_data])


        # Compute between-class scatter ( features x features )
        # ∑ Ni · (µi - µ) · (µi - µ)'
        between_class_scatter = sum([
            data.shape[0] * np.outer(mean - overall_mean, mean - overall_mean)
            for (data, mean) in zip(class_data, class_means)
        ])


        # Compute eigenvalues and eigenvectors of (Sw-1)Sb
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.inv(within_class_scatter) * between_class_scatter
        )

        # Compute optimal discriminants based on desired dimensions
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[sorted_indices]

        self.linear_discriminates = eigenvectors_sorted[:, :self.n_components]

    def transform(self, X):
        if self.linear_discriminates is None:
            raise Exception('LDA has not been fitted. Call the fit() method first.')
        else:
            return X @ self.linear_discriminates


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
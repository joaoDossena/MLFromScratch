import numpy as np

class PCA:
    

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance
        cov = np.cov(X.T)

        # Eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T # Transposing for easier calculations

        # Sort eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

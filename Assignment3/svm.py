from utils import *
import numpy as np
import cvxopt
import time
from joblib import Parallel, delayed


class SoftMarginSVMQP:

    def __init__(self, C=None, kernel='linear', gamma=1):

        '''
        Additional hyperparams allowed
        '''
        self.c = C
        self.kernel = kernel
        self.gamma = gamma

    def rbf_kernel(self, X, Z, gamma):

        if X.ndim == 1 and Z.ndim == 1:
            return np.exp((-np.linalg.norm(X - Z) ** 2) * gamma)
        elif (X.ndim > 1 and Z.ndim == 1) or (X.ndim == 1 and Z.ndim > 1):
            return np.exp((-np.linalg.norm(X - Z, axis=1) ** 2) * gamma)
        elif X.ndim > 1 and Z.ndim > 1:
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)
            sq_dists = X_norm + Z_norm - 2 * np.dot(X, Z.T)
            K = np.exp(gamma * sq_dists * -1)
            return K

    def linear_kernel(self, X, Z):
        if X.ndim == 1 and Z.ndim == 1:
            return np.dot(X, Z)
        else:
            return np.matmul(X, Z.T)

    def fit(self, X, y):
        '''
        TODO: Implement both lienar and RBF Kernels
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:

            None
        '''
        n_samples, n_features = X.shape
        K=None
        if self.kernel == 'rbf':
            K = self.rbf_kernel(X, X, self.gamma)
        if self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        # end of experimental code

        if self.c is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.diag(np.ones(n_samples)))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.c)))


        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y.astype(float), (1, n_samples))
        b = cvxopt.matrix(0.0)



        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        if self.kernel == 'linear':
            self.W = np.zeros(n_features)
            for n in range(len(self.a)):
                self.W += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.W = None

    def predict(self, X):
        '''
        TODO
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        Ouput:
            predictions : Shape : (no. of examples, )
        '''
        if self.kernel == 'linear':
            return np.where(X @ self.W + self.b >= 0, 1, -1)
        else:
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            sv_norm = np.sum(self.sv ** 2, axis=1).reshape(1, -1)
            dist_squared = X_norm + sv_norm - 2 * X @ self.sv.T

            K = np.exp(-self.gamma * dist_squared)

            y_predict = K @ (self.a * self.sv_y)
            return np.where(y_predict + self.b >= 0, 1, -1)
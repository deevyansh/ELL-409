from utils import *

class SoftMarginSVMQP:

    def __init__(self, C, kernel = 'linear'):

        '''
        Additional hyperparams allowed
        '''
        print(C,kernel)
        pass

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
        # below is an example
        self.W = np.ones(X.shape[1])
        self.b = 1

    def predict(self, X):
        '''
        TODO
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        Ouput:
            predictions : Shape : (no. of examples, )
        '''
        # below is an example
        return np.where(X @ self.W + self.b >= 0, 1 , -1)

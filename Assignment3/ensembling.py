from base_ensemble import *
from utils import *

class RandomForestClassifier(BaseEnsembler):

    def __init__(self, num_trees = 10):

        super().__init__(num_trees)

    def fit(self, X, y):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:

            None
        '''
        # below is an example
        pass

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
        val = np.random.randint(0,2,X.shape[0])
        val = np.where(val == 1, 1, -1)
        return val


class AdaBoostClassifier(BaseEnsembler):

    def __init__(self, num_trees = 10):

        super().__init__(num_trees)

    def fit(self, X, y):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes
        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:

            None
        '''
        # below is an example
        pass

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
        val = np.random.randint(0,2,X.shape[0])
        val = np.where(val == 1, 1, -1)
        return val

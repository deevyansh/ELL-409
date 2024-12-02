from base_ensemble import *
from utils import *
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

def entropy_cal(y):
    sum = np.sum(y)
    if (sum == 0 or sum == len(y)):
        return 0
    sum1 = len(y) - sum
    return -((sum / len(y)) * np.log10((sum / len(y)))) - ((sum1 / len(y)) * np.log10((sum1 / len(y))))


def gini_impurity(y):
    if (len(y) == 0):
        return 0
    sum = np.sum(y) / len(y)
    sum1 = (len(y) - np.sum(y)) / len(y)
    return (1 - np.power(sum, 2) - np.power(sum1, 2))


def information_gain(X, y, feature, threshold, option):
    mask = X[feature] <= threshold
    if option == "entropy":
        parent_entropy = entropy_cal(y)
        left_entropy = entropy_cal(y[mask])
        right_entropy = entropy_cal(y[~mask])
        weighted_entropy = (len(y[mask]) / len(y)) * left_entropy + (len(y[~mask]) / len(y)) * right_entropy
        return parent_entropy - weighted_entropy
    elif option == "gini":
        parent_gini = gini_impurity(y)
        left_gini = gini_impurity(y[mask])
        right_gini = gini_impurity(y[~mask])
        weighted_gini = (len(y[mask]) / len(y)) * left_gini + (len(y[~mask]) / len(y)) * right_gini
        return parent_gini - weighted_gini


class TreeNode:
    def __init__(self, X, y, entropy):
        self.X = X
        self.y = y
        self.entropy = entropy
        self.is_leaf_Node = False
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.prediction = 0
        self.count_terminal_Node = 0
        self.R_node = 0
        self.R_subtree = 0
        self.effective_alpha = 0
        self.cost = 0
        self.depth = 0

    def cal_prediction(self):
        if (np.sum(self.y) > (len(self.y) / 2)):
            self.prediction = 1
        else:
            self.prediction = -1

    def cal_risk(self, length):
        self.R_node = (min((np.sum(self.y)), (len(self.y) - np.sum(self.y))) / len(self.y)) * (len(self.y) / length)


class Decision_Tree:
    def __init__(self, depth, alpha, option):
        self.depth = depth
        self.alpha = alpha
        self.option = option

    def find_split(self, X, y):
        gain = float('-inf')
        feature = ""
        threshold = 0
        for column in X.columns:
            l = np.quantile(X[column].to_numpy(), np.linspace(0, 1, 100)[1:-1])
            for j in l:
                gain_value = information_gain(X, y, column, j, self.option)
                if (gain < gain_value):
                    feature = column
                    threshold = j
                    gain = gain_value
        return (feature, threshold)

    def give_child(self, X, y, feature, threshold):
        mask = (X[feature] <= threshold)
        if mask.sum() == 0 or (~mask).sum() == 0:
            return None, None  # No valid split
        left = TreeNode(X[mask], y[mask], entropy_cal(y[mask]))
        right = TreeNode(X[~mask], y[~mask], entropy_cal(y[~mask]))
        return left, right

    def make_tree(self, Node, depth, length):
        if (depth > self.depth):
            Node.is_leaf_Node = True
            Node.cal_prediction()
            return
        else:
            feature, threshold = self.find_split(Node.X, Node.y)
            left, right = self.give_child(Node.X, Node.y, feature, threshold)
            if (left == None):
                Node.is_leaf_Node = True
                Node.cal_prediction()
            else:
                Node.left = left
                Node.right = right
                Node.feature = feature
                Node.threshold = threshold
                self.make_tree(Node.left, depth + 1, length)
                self.make_tree(Node.right, depth + 1, length)
        Node.cal_risk(length)

    def prediction(self, Node, X):
        if (Node.is_leaf_Node):
            return Node.prediction
        else:
            if (X[Node.feature] <= Node.threshold):
                return self.prediction(Node.left, X)
            else:
                return self.prediction(Node.right, X)

    def accuracy(self, X_val, y_val, Node):
        pos_count = 0
        X_val = X_val.reset_index(drop=True)
        for i in range(len(X_val)):
            pos_count += (y_val[i] == self.prediction(Node, X_val.iloc[i]))
        return pos_count / len(y_val)

    def model_evaluate(self, X_val, y_val, Node):
        corr_pos_count = 0
        pos_count = 0
        X_val = X_val.reset_index(drop=True)
        for i in range(len(X_val)):
            if ((self.prediction(Node, X_val.iloc[i]) == True) and y_val[i] == 1):
                corr_pos_count += 1
            pos_count += (self.prediction(Node, X_val.iloc[i]) == 1)
        recall = corr_pos_count / (np.sum(y_val))
        precision = (corr_pos_count) / (pos_count)
        f_score = (recall * precision * 2) / (recall + precision)
        return recall, precision, f_score

    def fit(self, Node):
        if (Node == None):
            return
        if (Node.is_leaf_Node):
            Node.cal_prediction()
            Node.R_subtree = Node.R_node
            Node.count_terminal_Node = 1
            Node.effective_alpha = float('inf')
            Node.depth = 1
        else:
            self.fit(Node.left)
            self.fit(Node.right)
            Node.cal_prediction()
            Node.count_terminal_Node = Node.left.count_terminal_Node + Node.right.count_terminal_Node
            Node.R_subtree = Node.left.R_subtree + Node.right.R_subtree
            Node.effective_alpha = (Node.R_node - Node.R_subtree) / (Node.count_terminal_Node - 1)
            Node.depth = max(Node.right.depth, Node.left.depth) + 1

    def prune(self, Node, alpha):
        if (Node.is_leaf_Node):
            return
        if (Node.effective_alpha < alpha):
            Node.is_leaf_Node = True
            Node.left = None
            Node.right = None
        else:
            self.prune(Node.left, alpha)
            self.prune(Node.right, alpha)



class RandomForestClassifier(BaseEnsembler):

    def __init__(self, num_trees , max_depth):
        super().__init__(num_trees)
        self.max_depth=max_depth
        self.decision_trees=[]
        self.num_trees=num_trees
        self.node_list=[]

    def randomize(self,X,y):
        indices=np.random.choice(len(X), len(X))
        X_temp=X[indices]
        y_temp = np.where(y[indices] == -1, 0, 1)
        X_temp = pd.DataFrame(X_temp, columns=[f"feature_{i}" for i in range(X.shape[1])])
        # columns = np.random.choice(X_temp.columns, self.num_feature, replace=False)
        # X_temp = X_temp[columns]
        return X_temp, pd.Series(y_temp)


    def fit(self, X, y):
        def train_tree(i, X, y, max_depth):
            print(i, "Processing")
            X_temp, y_temp = self.randomize(X, y)  # Assuming `randomize` is defined elsewhere
            Node = TreeNode(X_temp, y_temp, entropy_cal(y_temp))
            decision_tree_temp = Decision_Tree(max_depth, 2, "entropy")
            decision_tree_temp.make_tree(Node, 1, len(X_temp))
            return decision_tree_temp, Node

        results = Parallel(n_jobs=20)(
            delayed(train_tree)(i, X, y, self.max_depth) for i in range(self.num_trees)
        )
        self.decision_trees = [result[0] for result in results]
        self.node_list = [result[1] for result in results]
            

    def predict(self, X):
        prediction=[]
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        for j in range (X.shape[0]):
            total_votes=0
            for i in range (len(self.decision_trees)):
                total_votes+=self.decision_trees[i].prediction(self.node_list[i],X.loc[j])
            if(total_votes>=0):
                prediction.append(1)
            else:
                prediction.append(-1)
        return pd.Series(prediction)






class AdaBoostClassifier(BaseEnsembler):

    def __init__(self, num_trees = 10):
        super().__init__(num_trees)
        self.decision_trees=[]
        self.amount_of_says=[]
        self.num_trees=num_trees
        self.node_list=[]

    def fit(self, X, y):
        weights = np.ones(len(X)) / len(X)
        self.decision_trees=[]
        self.amount_of_says=[]
        self.node_list=[]
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        for i in range (0,self.num_trees):
            print(i,"Processing")
            chosen_indices = np.random.choice(len(X), size=len(X), replace=True, p=weights)
            X_temp=X[chosen_indices]
            y_temp = np.where(y[chosen_indices] == -1, 0, 1)
            X_temp = pd.DataFrame(X_temp, columns=[f"feature_{i}" for i in range(X.shape[1])])
            decision_Tree=Decision_Tree(1,1,"gini")
            Node=TreeNode(X_temp,y_temp,entropy_cal(y_temp))
            decision_Tree.make_tree(Node,1,len(X_temp))
            self.decision_trees.append(decision_Tree)
            self.node_list.append(Node)

            predictions = np.array([decision_Tree.prediction(Node, X_df.iloc[j]) for j in range(len(X))])
            incorrect=(predictions!=y)
            error=np.sum(weights[incorrect])

            total_amount_of_say=0.5 * np.log((1 - error) / error)
            self.amount_of_says.append(total_amount_of_say)
            weights *= np.exp(-y * predictions * total_amount_of_say)
            weights /= np.sum(weights)
            
        
    def predict(self, X):
        # below is an example
        prediction=[]
        X=pd.DataFrame(X,columns=[f"feature_{i}" for i in range(X.shape[1])])
        for i in range (len(X)):
            sum1=0
            for j in range (len(self.decision_trees)):
                    sum1+=self.amount_of_says[j]*self.decision_trees[j].prediction(self.node_list[j],X.iloc[i])
            if(sum1>0):
                prediction.append(1)
            else:
                prediction.append(-1)
        return pd.Series(prediction)

    def validation(self,X,y, val_X, val_y):
        validation_score=[]
        config=[]
        weights = np.ones(len(X)) / len(X)
        self.decision_trees = []
        self.amount_of_says = []
        self.node_list = []
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        for i in range(0, self.num_trees):
            config.append(i)
            print(i, "Processing")
            chosen_indices = np.random.choice(len(X), size=len(X), replace=True, p=weights)
            X_temp = X[chosen_indices]
            y_temp = np.where(y[chosen_indices] == -1, 0, 1)
            X_temp = pd.DataFrame(X_temp, columns=[f"feature_{i}" for i in range(X.shape[1])])
            decision_Tree = Decision_Tree(1, 1, "gini")
            Node = TreeNode(X_temp, y_temp, entropy_cal(y_temp))
            decision_Tree.make_tree(Node, 1, len(X_temp))
            self.decision_trees.append(decision_Tree)
            self.node_list.append(Node)

            predictions = np.array([decision_Tree.prediction(Node, X_df.iloc[j]) for j in range(len(X))])
            incorrect = (predictions != y)
            error = np.sum(weights[incorrect])

            total_amount_of_say = 0.5 * np.log((1 - error) / error)
            self.amount_of_says.append(total_amount_of_say)
            weights *= np.exp(-y * predictions * total_amount_of_say)
            weights /= np.sum(weights)

            validation_score.append(val_score(self.predict(val_X), val_y))

        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.plot(config,validation_score)
        plt.savefig("adaboost_validation.png")
        ax.set_xlabel("Number of trees")
        ax.set_ylabel("Validation F1-score")
        plt.show()
        print(validation_score,config)



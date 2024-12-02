

## Analysis for the SVM ##















### Analysis for the Random Forest###

from utils import *
from decision_tree import *
from ensembling import *
from config import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random

train_processor = MNISTPreprocessor('./dataset/train', PRE_PROCESSING_CONFIG)
train_X, train_y = train_processor.get_all_data()
train_X, train_y = train_processor.filter_dataset(train_X,train_y, ENTRY_NUMBER_LAST_DIGIT)
train_y = convert_labels_to_svm_labels(train_y, ENTRY_NUMBER_LAST_DIGIT)
train_dict = {"X":train_X, "y": train_y}

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
train_X = pca.fit_transform(train_dict["X"])

val_processor = MNISTPreprocessor('./dataset/val', PRE_PROCESSING_CONFIG)
val_X,val_y = val_processor.get_all_data()
val_X,val_y = val_processor.filter_dataset(val_X,val_y, ENTRY_NUMBER_LAST_DIGIT)
val_y = convert_labels_to_svm_labels(val_y, ENTRY_NUMBER_LAST_DIGIT)
val_X = pca.fit_transform(val_X)

max_depth=10
max_trees=10

def validation(train_X, train_y, val_X, val_y, max_depth, max_trees):
    configs=[]
    f_score=[]

    def train_and_evaluate(i, j):
        model = RandomForestClassifier(i,10,j)
        model.fit(train_X, train_y)
        score = val_score(val_y, model.predict(val_X))
        return [i, j], score

    # Use Parallel to run the train_and_evaluate function across different configurations
    results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(i, j)
                                  for i in range(3, max_depth)
                                  for j in range(3, max_trees))

    # Extract configurations and scores from results
    for config, score in results:
        configs.append(config)
        f_score.append(score)
    print(configs)
    print(f_score)
    configs = np.array(configs)
    f_score = np.array(f_score)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(configs[:, 0], configs[:, 1], f_score, c=f_score, cmap='viridis', marker='o')

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Number of Trees')
    ax.set_zlabel('F-Score')
    print(max(f_score))
    plt.savefig("validation.png")


validation(train_X, train_y, val_X,val_y,10,10)


def random_validation(train_X, train_y, val_X, val_y, max_depth, max_trees):
    configs=[]
    f_score=[]
    for _ in range(10):
        i = np.random.randint(3, max_depth)
        j = random.randint(3, max_trees)
        model = RandomForestClassifier(i, 10, j)
        model.fit(train_X, train_y)
        f_score.append(val_score(val_y, model.predict(val_X)))
        configs.append([i, j])

    configs = np.array(configs)
    f_score = np.array(f_score)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(configs[:, 0], configs[:, 1], f_score, c=f_score, cmap='viridis', marker='o')

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Number of Trees')
    ax.set_zlabel('F-Score')
    plt.savefig("random_validation.png")

random_validation(train_X, train_y, val_X, val_y, 10,10)


### Analysis for the Ada Boost###



from ensembling import *
from config import *
from utils import *

train_processor = MNISTPreprocessor('./dataset/train', PRE_PROCESSING_CONFIG)
train_X, train_y = train_processor.get_all_data()
train_X, train_y = train_processor.filter_dataset(train_X,train_y, ENTRY_NUMBER_LAST_DIGIT)
train_y = convert_labels_to_svm_labels(train_y, ENTRY_NUMBER_LAST_DIGIT)
train_dict = {"X":train_X, "y": train_y}

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
train_X = pca.fit_transform(train_dict["X"])
#
val_processor = MNISTPreprocessor('./dataset/val', PRE_PROCESSING_CONFIG)
val_X,val_y = val_processor.get_all_data()
val_X,val_y = val_processor.filter_dataset(val_X,val_y, ENTRY_NUMBER_LAST_DIGIT)
val_y = convert_labels_to_svm_labels(val_y, ENTRY_NUMBER_LAST_DIGIT)
val_X = pca.fit_transform(val_X)
# #
# model=AdaBoostClassifier(60)
# model.validation(train_X, train_y,val_X,val_y)


def random_validation(train_X, train_y, val_X, val_y, max_trees):
    validation_score=[]
    configs=[]
    for _ in range(10):
        i = np.random.randint(3, max_trees)
        model=AdaBoostClassifier(i)
        model.fit(train_X,train_y)
        validation_score.append(val_score(model.predict(val_X), val_y))
        configs.append(i)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(configs, validation_score)
    plt.savefig("adaboost_validation.png")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Validation F1-score")
    plt.show()

random_validation(train_X,train_y,val_X,val_y,30)
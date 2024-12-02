ENTRY_NUMBER_LAST_DIGIT = 6 # change with yours
ENTRY_NUMBER = '2021MT10236'
PRE_PROCESSING_CONFIG ={
    "hard_margin_linear" : {
        "use_pca" : None,
    },

    "hard_margin_rbf" : {
        "use_pca" : None,
    },

    "soft_margin_linear" : {
        "use_pca" : None,
    },

    "soft_margin_rbf" : {
        "use_pca" : None,
    },

    "AdaBoost" : {
        "use_pca" : None,
    },

    "RandomForest" : {
        "use_pca" : None,
    }
}

SVM_CONFIG = {
    "hard_margin_linear" : {
        "C" : 1e9,
        "kernel" : 'linear',
        "val_score" : 0, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },
    "hard_margin_rbf" : {
        "C" : 1e9,
        "kernel" : 'rbf',
        "val_score" : 0, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_linear" : {
        "C" : 1, # add your best hyperparameter
        "kernel" : 'linear',
        "val_score" : 0, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_rbf" : {
         "C" : 1, # add your best hyperparameter
         "kernel" : 'rbf',
         "val_score" : 0, # add the validation score you get on val set for the set hyperparams.
                          # Diff in your and our calculated score will results in severe penalites
         # add implementation specific hyperparams below (with one line explanation)
     }
}

ENSEMBLING_CONFIG = {
    'AdaBoost':{
        'num_trees' : 10,
        "val_score" : 0,
    },

    'RandomForest':{
        'num_trees' : 10,
        "val_score" : 0,
    }
}

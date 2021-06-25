# Load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import roc_auc_score,roc_curve

# Settings
dirname_PKL='PKLfiles'

# Purpose: Creates a NumPy input data matrix and output label vector
# Arguments: df: A DataFrame, the input dataset.
#            feature_names: A list of strings, where each string is a feature of interest from x1.
#            out_var: A string, the name of the output variable
# Returns: A tuple, consisting of a NumPy data matrix, the input data, and a NumPy array, the output labels
def get_np(df, feature_names, out_var):
    dic = {f:df[f] for f in feature_names}
    data_np = np.array([dic[i] for i in dic]).T
    label = np.array(df[out_var], dtype=np.uint8)
    return (data_np, label)

# Purpose: Load saved preprocessed features and labels from CSV files
# Arguments: str_file_postfix: A string, postfix file name saved.
# Returns: A tuple of following values:
#          - A NumPy array, training features.
#          - A NumPy array, testing features.
#          - A NumPy array, normalized training features.
#          - A NumPy array, normalized testing features.
#          - A NumPy array, training labels.
#          - A NumPy array, testing labels.
#          - A list of strings, where each string is a feature of interest.
#          - A string, the name of the output variable
def load_processed_features(str_file_postfix):
    str_filename_train_prefix     ='train_cac_prep_data'
    str_filename_test_prefix      ='test_cac_prep_data'
    str_filename_train_norm_prefix='train_cac_prep_norm_data'
    str_filename_test_norm_prefix ='test_cac_prep_norm_data'

    train_dataframe_label     =pd.read_csv(str_filename_train_prefix+str_file_postfix, low_memory=False, na_values='#NULL!')
    test_dataframe_label      =pd.read_csv(str_filename_test_prefix+str_file_postfix, low_memory=False, na_values='#NULL!')
    train_dataframe_norm_label=pd.read_csv(str_filename_train_norm_prefix+str_file_postfix, low_memory=False, 
                                           na_values='#NULL!')
    test_dataframe_norm_label =pd.read_csv(str_filename_test_norm_prefix+str_file_postfix, low_memory=False, na_values='#NULL!')

    feature_names=list(train_dataframe_label.columns[:-1])
    out_var=str(train_dataframe_label.columns[-1])
    
    train_data, train_label     =get_np(train_dataframe_label, feature_names, out_var)
    test_data, test_label       =get_np(test_dataframe_label, feature_names, out_var)
    train_data_norm, train_label=get_np(train_dataframe_norm_label, feature_names, out_var)
    test_data_norm, test_label  =get_np(test_dataframe_norm_label, feature_names, out_var)

    return (train_data, test_data, train_data_norm, test_data_norm, train_label, test_label, feature_names, out_var)

# Purpose: Set up the model. For the model, initialize and tune parameters with cross validation.
# Arguments: train_data: A NumPy 2-D array containing the input training data for the model.
#            train_label: A NumPy 1-D array of ints, the output training labels for the model.
#            seed_random: An int, the random seed for reproducibility.
# Returns: The model with tuned parameters.
def setup_classifier_training(train_data, train_label, seed_random):
    # Set seed if not none
    if seed_random is not None:
        np.random.seed(seed_random)

    # Setup cross validation split
    skf = StratifiedKFold(n_splits=5)
    index_train_test_cv = []
    for index_train in skf.split(train_data, train_label):
        index_train_test_cv.append(index_train)
        
    # Setup CV hyperparameter search
    est = XGBClassifier(max_iter=250)
    param = [{'max_depth': np.arange(1, 9, 1)}]
    model = GridSearchCV(estimator = est, param_grid = param,
                              scoring='roc_auc', cv = index_train_test_cv)
    return model


# Purpose: Fits the model, with its tuned parameters, to the training data.
# Arguments: train_data: A NumPy 2-D array containing the input training data for the model.
#            train_label: A NumPy 1-D array of ints, the output training labels for the model.
#            model: A model with tuned parameters.
#            model_post_filename: A string, the pickle file name.
#            algo_name: A string, the algorithm name.
#            toggle_perform_training: An int, 0 or 1. 1 indicates the model needs to be trained. 0 indicates the model
#                                   does not need to be trained. (Defaults to 1).
#            toggle_save_trained_model: An int, 0 or 1. 1 indicates the model needs to be saved. 0 indicates the model
#                                    does not need to be saved. (Defaults to 1).
# Returns: A fitted model.
def fit_mod(train_data,  train_label, model, model_post_filename, algo_name='',
            toggle_perform_training=1, toggle_save_trained_model=1):
    if not toggle_perform_training:
        model = None

    str_save_model_PKL = dirname_PKL + '/' + 'SavedModel_' + model_post_filename + '_' +  algo_name + '_00.pkl'
    if toggle_perform_training:
        # Fit the model to the training data
        model.fit(train_data, train_label)
        if toggle_save_trained_model:
            pickle.dump(model, open(str_save_model_PKL, 'wb'))
            print('Saved as:', str_save_model_PKL)
    else:
        model = pickle.load(open(str_save_model_PKL, 'rb'))
        print('Loaded file:', str_save_model_PKL)
                
    return model

# Purpose: Gets the classifier metrics for the model.
# Arguments: model: The fitted model.
#            test_data: A NumPy 2-D array containing the input testing data for the model.
#            test_label: A NumPy 1-D array of ints, the output testing labels for the model.
#            classifier_threshold: A float, the classifier threshold.
# Returns: A tuple of following values:
#          - A float, the roc auc score.
#          - A list of two floats, the false positive rate and the true positive rate (for the roc curve).
#          - An int, the number of positives the model predicts correctly.
#          - An int, the number of positives the model predicts incorrectly.
#          - An int, the number of negatives the model predicts incorrectly.
#          - An int, the number of negatives the model predicts correctly.
def get_roc_auc_values(model, test_data, test_label, classifier_threshold):
    
    test_prob = np.array(model.predict_proba(test_data)[:, 1], dtype = float)
    roc_auc = roc_auc_score(test_label, test_prob)
    fpr, tpr,_ = roc_curve(test_label, test_prob)
    roc_curve_values = [fpr, tpr]
    
    test_pred = np.array(test_prob >= classifier_threshold, dtype = int)
    TP = np.sum(np.logical_and(test_pred == 1, test_label == 1))
    FP = np.sum(np.logical_and(test_pred == 1, test_label == 0))
    FN = np.sum(np.logical_and(test_pred == 0, test_label == 1))
    TN = np.sum(np.logical_and(test_pred == 0, test_label == 0))

    return (roc_auc, roc_curve_values, TP, FP, FN, TN)
 
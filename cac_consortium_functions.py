# Load libraries
import pandas as pd
pd.options.mode.chained_assignment = None # Avoid setting with copy warning 
                                          # for create a rowid column
import numpy as np
import matplotlib.pyplot as plt
import os
import string

from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import sklearn.svm as svm

from utils import *
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost
from xgboost import XGBClassifier

import pickle

from timeit import default_timer as timer

from sklearn.metrics import roc_auc_score,roc_curve
from auc_delong_xu import auc_ci_Delong

# Settings
dirnamePKL='PKLfiles'
toggleNumInstanceRuns=10

# Functions

# Purpose: Reads and structures the data.
# Arguments: None.
# Returns: A tuple consisting of a DataFrame, the structured data, and a 
#          string, the name of the output variable.
def read_and_structure():
    # Read Data, rename death column, make death column a categorical variable
    out_var='Deathtenytruncated'
    cac_data = pd.read_csv('EHJ_cohort.csv', low_memory=False)
    cac_data.rename(columns = {"Death10ytruncated": "Deathtenytruncated"}, 
                    inplace = True)
    cac_data["Deathtenytruncated"] = \
    cac_data["Deathtenytruncated"].astype("category")
    
    # Create a row id column
    cac_data["index"] = cac_data.index
    
    # Select variables of interest
    simple_cac_data = cac_data[["index", "age", "total_score", 
                                "Deathtenytruncated"]]
    
    return simple_cac_data, out_var

# Purpose: Splits the data into 2 sets: training and testing.
# Arguments: cac_data: The structured DataFrame.
#            seedRandom: An int, the random seed for reproducibility. Defaults
#                        to None.
# Returns: A tuple of two DataFrames, the training and the testing data.
def split_data(cac_data, seedRandom = None):
    # Eighty-Twenty split with training being eighty percent
    # and testing being twenty percent
    training_cac_indices = None
    testing_cac_indices = None
    
    sss = StratifiedKFold(n_splits = 5, shuffle = True, 
                          random_state = seedRandom)
    (training_cac_indices, testing_cac_indices) = \
        next(sss.split(cac_data[["index", "age", "total_score"]], 
                       cac_data[["Deathtenytruncated"]]))

    training_cac_data = cac_data.loc[training_cac_indices, :]
    training_cac_data.reset_index(inplace = True)
    training_cac_data = training_cac_data[["index", "age", "total_score", 
                                           "Deathtenytruncated"]]
    
    testing_cac_data = cac_data.loc[testing_cac_indices, :]
    testing_cac_data.reset_index(inplace = True)
    testing_cac_data = testing_cac_data[["index", "age", "total_score", 
                                         "Deathtenytruncated"]]
    
    return (training_cac_data, testing_cac_data)

# Purpose: Check data splits for correct subsampled indices
# Arguments: listNameSplit: list of names of splits such as 
#                           ['Original', 'Train', 'Test']
#            listDataSplit: list of DataFrames, where each DataFrame 
#                           corresponds to the name in listNameSplit, and the 
#                           first DataFrame is the original complete dataset.
# Returns: n/a (only prints)
def check_subsampled_split_data(listNameSplit, listDataSplit):    
    # Check indices
    indSplit=1
    numPrintRow=3
    print('Headings:', list(listDataSplit[0].keys()))
    print()
    for dataSplit in listDataSplit[1:]:
        print('----------------------')
        print('Match Indices {} in {}'.format(listNameSplit[indSplit],
                                              listNameSplit[0]))
        print('----------------------')
        valsSource = np.array(dataSplit)[:, 1:]
        valsDest = np.array(listDataSplit[0])[dataSplit['index'], 1:]
        vecMatch = np.sum(valsSource == valsDest, axis = 1) == 3
        numMatch = np.sum(vecMatch)
        numNoMatch = len(dataSplit) - numMatch
        indNoMatch = np.where(vecMatch == 0)[0]
        print('Match={:.2f}% ({}), NoMatch={:.2f}% ({}), Size={}'.format(
            numMatch / len(dataSplit) * 100, numMatch,
            numNoMatch / len(dataSplit) * 100,
            numNoMatch, len(dataSplit)))

        if numNoMatch > 0:
            rangeSource = indNoMatch[0:np.minimum(numPrintRow, numNoMatch)]
        else: # numNoMatch==0
            rangeSource = range(numPrintRow)
        for indSource in rangeSource:
            rowSource = dataSplit.iloc[indSource]
            indDest = rowSource['index'].astype(int)
            rowDest = listDataSplit[0].iloc[indDest]
            toPrint = 'Split[{}]:idx={:5}, age={:.2f}, score={:4.0f},' + \
                      'death={:.0f}-> Orig[{:5}]:age={:.2f}, score={:4.0f},' +\
                      'death={:.0f}| Match={}'
            print(toPrint.format(indSource, indDest, rowSource['age'], 
                                 rowSource['total_score'],
                                 rowSource['Deathtenytruncated'],
                                 indDest, rowDest['age'],
                                 rowDest['total_score'],
                                 rowDest['Deathtenytruncated'],
                                 vecMatch[indSource]))
        indSplit += 1
    print()
   
# Purpose: Check data splits for stratification.
# Arguments: listNameSplit: list of names of splits such as 
#                           ['Original', 'Train', 'Test']
#            listDataSplit: list of DataFrames, where each DataFrame 
#                           corresponds to the name in listNameSplit, and the 
#                           first DataFrame is the original complete dataset.
# Returns: n/a (only prints)
def check_stratified_split_data(listNameSplit, listDataSplit):        
    print('----------------------')
    print('Verify Stratification')
    print('----------------------')

    # Data Split
    print('{:20}'.format('Split Data:'), end = '')
    for value in listNameSplit: 
        print('{:>14}'.format(value), end = '')
    print()

    # Size Percent
    arrSize = np.array([ len(item) for item in listDataSplit ])
    print('{:20}'.format('Size:'), end = '')
    for value in arrSize: 
        print('{:14}'.format(value), end = '')
    print()
    arrSizePct = arrSize / arrSize[0] * 100
    print('{:20}'.format('Size Percent:'), end = '')
    for value in arrSizePct:
        print('{:13.1f}%'.format(value), end = '')
    print()
    print()

    # Death Prevalence
    strLabel = 'Deathtenytruncated'
    arrLabelCount = np.array([ np.sum(item[strLabel] == 1) for item in \
                              listDataSplit ])
    arrPrevPct = arrLabelCount / arrSize * 100
    print('{:20}'.format('Death Prevalence:'), end = '')
    for value in arrPrevPct: 
        print('{:13.3f}%'.format(value), end = '')
    print()
    
    # Chi-squared test assumes categorical with all frequencies above 5 
    arrLabelPval = np.array([ stats.chisquare([freq, 100 - freq],
                                              f_exp = [arrPrevPct[0],
                                              100 - arrPrevPct[0]])[1] \
                            for freq in arrPrevPct ])
    print('{:20}'.format('Death (chi-sq) p:'), end = '')
    for value in arrLabelPval:
        print('{:14.4g}'.format(value), end = '')
    print()
    print()

    strLabel='age'
    arrAgeMean = np.array([ np.mean(item[strLabel]) for item in \
                            listDataSplit ])
    arrAgeStd = np.array([ np.std( item[strLabel]) for item in listDataSplit ])
    print('{:20}'.format('Age Mean ±Std:'), end = '')
    for values in zip(arrAgeMean, arrAgeStd): 
        print('{:7.1f} ±{:5.1f}'.format(values[0], values[1]), end = '')
    print()
    
    # t-test indepedent assume normal and continuous (interval)
    arrAgePval = np.array([ stats.ttest_ind(listDataSplit[0][strLabel], 
                                            item[strLabel])[1] \
                            for item in listDataSplit ])
    print('{:20}'.format('Age (t-test) p:'), end = '')
    for value in arrAgePval:
        print('{:14.4g}'.format(value), end = '')
    print()
    print()

    # Total Score Metrics
    strLabel='total_score'
    
    # Total Score Median
    arrScoreMedian = np.array([ np.median(item[strLabel]) for item in \
                                listDataSplit ])
    arrScoreQ1 = np.array([ np.quantile(item[strLabel], .25) for item in \
                            listDataSplit ])
    arrScoreQ3 = np.array([ np.quantile(item[strLabel], .75) for item in \
                            listDataSplit ])
    print('{:20}'.format('Score Median [IQR]:'), end = '')
    for values in zip(arrScoreMedian, arrScoreQ1, arrScoreQ3): 
        print('{:5.1f} [{:2.0f},{:3.0f}]'.format(values[0],
                                                 values[1],
                                                 values[2]), end = '')
    print()
    
    # Total Score Mean
    arrScoreMean = np.array([ np.mean(item[strLabel]) for item in \
                              listDataSplit ])
    arrScoreStd = np.array([ np.std( item[strLabel]) for item in \
                             listDataSplit ])
    print('{:20}'.format('Score Mean ±Std:'), end = '')
    for values in zip(arrScoreMean, arrScoreStd):
        print('{:7.1f} ±{:5.1f}'.format(values[0], values[1]), end = '')
    print()
    
    # Two-sample Wilcoxon rank-sum (Mann-Whitney) test does not assume normal 
    # but continuous (interval)
    arrScorePval = np.array([ stats.ranksums(listDataSplit[0][strLabel], 
                                             item[strLabel])[1] \
                              for item in listDataSplit ])
    print('{:20}'.format('Score (rank sum) p:'), end = '')
    for value in arrScorePval: 
        print('{:14.4g}'.format(value), end = '')
    print()
    print()    
    
# Purpose: Generates multiple train-validation data splits for stratified k 
#          fold cross validation.
# Arguments: data_file: a DataFrame, the dataset to be split.
#            out_var: a string, the output variable name.
#            seedRandom: An int, the random seed for reproducibility.
#            numSplits: An int, the number of folds for stratified k fold
#                       cross validation. Defaults to 5 (80-20 split).
#            toggleForceGenerateNewInstanceRuns: an int, 1 to generate new 
#                                                splits and 0 to generate old 
#                                                splits. Defaults to zero.
# Returns: A tuple consisting of 2 lists of DataFrames. The first list contains
#          k DataFrames, which is the training data, and the second list
#          contains k DataFrames, which is the testing data.
def train_validation_data_split(data_file, out_var, seedRandom, numSplits = 5, 
                                toggleForceGenerateNewInstanceRuns = 0):

    # Set seed if not none
    if seedRandom is not None:
        np.random.seed(seedRandom)    

    feature_names = [i for i in data_file.keys() if i not in [out_var]]
    
    dic_features = {i:np.array(data_file[i]) for i in feature_names}
    data_np = np.array([dic_features[i] for i in dic_features]).T 
    label = np.array(data_file[out_var])
    
    skf = StratifiedKFold(n_splits = numSplits, shuffle = True)

    listDataFileTrain = []
    listDataFileTest  = []    
    for indRun, (train_idx, test_idx) in enumerate(skf.split(data_np, label)):
        strFileTrain = dirnamePKL + '\\' + 'train_idx_' + \
                       '{0:03d}'.format(indRun + 1) + '.pkl'
        if toggleForceGenerateNewInstanceRuns or not \
           os.path.exists(strFileTrain):
            pickle.dump(train_idx, open(strFileTrain, 'wb'))
            print('Wrote', strFileTrain)
        else:
            train_idx = pickle.load(open(strFileTrain, 'rb'))
            print('Already exists. Read', strFileTrain)
        train_data_file = data_file.loc[train_idx] 
        listDataFileTrain.append(train_data_file)
        
        strFileTest = dirnamePKL + '\\' + 'test_idx_' + \
                      '{0:03d}'.format(indRun + 1) + '.pkl'
        if toggleForceGenerateNewInstanceRuns or not \
           os.path.exists(strFileTest):
            pickle.dump(test_idx, open(strFileTest, 'wb'))
            print('Wrote', strFileTest)
        else:
            test_idx  = pickle.load(open(strFileTest, 'rb'))
            print('Already exists. Read', strFileTrain)
        test_data_file  = data_file.loc[test_idx]
        listDataFileTest.append(test_data_file)    
         
    return (listDataFileTrain, listDataFileTest)

# Purpose: Balances the training cac data.
# Arguments: cac_data: A DataFrame, the training cac data.
#            death_ratio: A float, the ratio of nondeaths to deaths
#            seedRandom: An int, the random seed for reproducibility. Defaults
#                        to None.
# Returns: A DataFrame, the balanced training cac data.
def balance_training_data(cac_data, death_ratio, out_var, seedRandom = None):
    # Set seed if not none
    if seedRandom is not None:
        np.random.seed(seedRandom)

    # Create a DataFrame with all the deaths in the training cac data and
    # create a DataFrame with all the nondeaths in the training cac data
    cac_data_deaths = cac_data[cac_data[out_var] == 1]
    cac_data_nondeaths = cac_data[cac_data[out_var] == 0]

    # Get number of deaths in the training cac data
    number_of_cac_data_deaths = len(cac_data_deaths.index)

    # Select a subset of the nondeaths from the training cac data
    # The number of nondeaths selected is equal to the number of the deaths
    # in the training cac data multiplid by the ratio of nondeaths to deaths
    selected_cac_data_nondeaths_indices = \
        np.random.choice(list(cac_data_nondeaths.index),
                         int(number_of_cac_data_deaths * death_ratio))
    selected_cac_data_nondeaths = \
        cac_data_nondeaths.loc[selected_cac_data_nondeaths_indices, :]

    # Concatenate the subset of nondeaths to the deaths to create the
    # balanced training cac data
    balanced_cac_data  = pd.concat([cac_data_deaths,
                                    selected_cac_data_nondeaths])

    return balanced_cac_data

# Purpose: Creates a NumPy input data matrix and output label vector
# Arguments: x1: A DataFrame, the input dataset.
#            f_names: A list of strings, where each string is a feature of
#                     interest from x1.
#            out_var: A string, the name of the output variable
# Returns: A tuple, consisting of a NumPy data matrix, the input data, and a
#          NumPy array, the output labels
def get_np(x1, f_names, out_var):
    dic = {i:x1[i] for i in f_names}
    data_np = np.array([dic[i] for i in dic]).T
    label = np.array(x1[out_var], dtype=np.uint8)
    return (data_np, label)

# Purpose: Normalize data by mean and std
# Arguments: data: a NumPy input data matrix
#            f_names: A list of strings, where each string is a feature of
#                     interest from x1.
#            mean_dic: A dictionary that contains the mean for each
#                      feature of the input data matrix.
#            std_dic: A dictionary that contains the standard deviation for
# #                   each feature of the input data matrix.
# Returns: A NumPy data matrix, the normalized data.
def norm(data, feature_names, mean_dic, std_dic):
    data_norm = data.copy()
    for i in mean_dic:
        data_norm[:, feature_names.index(i)] = \
            (data_norm[:, feature_names.index(i)] - mean_dic[i]) / std_dic[i]
    return data_norm

# Purpose: Prepare data for the model (xgboost).
# Arguments: train_data_file: A DataFrame, the training data for the model.
#            test_data_file: A DataFrame, the testing data for the model.
#            out_var: A string, the output variable name.
#            togglePrintNormalization: An int, one to print the variable
#                                      normalizations and zero to not print the
#                                      normalizations. Defaults to zero.
#            toggleInteraction: An int, one to include the interaction term
#                               between age and total score, and zero to
#                               exclude the interaction term. Defaults to one.
# Returns: A tuple containing eleven items:
#          - A string, the model name.
#          - A string, the pickle file name.
#          - A list of strings, containing the feature names.
#          - A NumPy 2-D array, containing the normalized input training
#            data for the model.
#          - A NumPy 2-D array, containing the normalized input testing
#            data for the model.
#          - A NumPy 2-D array, containing the unnormalized input training
#            data for the model.
#          - A NumPy 2-D array, containing the unnormalized input testing
#            data for the model.
#          - A NumPy 1-D array of ints, the output training labels for
#            the model.
#          - A NumPy 1-D array of ints, the output testing labels for
#            the model.
#          - A dictionary, which contains the model's continuous variables as
#            the keys and the variables' corresponding means as the values.
#          - A dictionary, which contains the model's continuous variables as
#            the keys and the variables' corresponding standard deviations as
#            the values.
def prep_mod_data(train_data_file, test_data_file, out_var,
                  togglePrintNormalization = 0, toggleInteraction = 1):
    modelName = 'Age+Score'
    modelPostFilename = modelName.translate(str.maketrans('', '',
                                            string.punctuation + ' '))

    toggleAgeASOnly = 1
    if toggleAgeASOnly:
        if toggleInteraction:
            listFeatureCommon = ['age', 'total_score', 'total_score_x_age']
        else:
            listFeatureCommon = ['age', 'total_score']
    else:
        # FIX!!! - check if smokegroup (0,1,2,3), race_recat (1,2,3,4,5,6)
        # needs to be changed to categorical
        toggleIncludeFamilyHistory = 1
        if toggleIncludeFamilyHistory:
            listFeatureCommon = ['study_site', 'age', 'sex',
                                 'hypertension_nonimpute', 'diabetes_nonimpute',
                                 'familyhistory_nonimpute',
                                 'hyperlipidemia_nonimpute',
                                 'smokegroup_recalc', 'race_recat',
                                 'numberofvessels', 'lm_score', 'lad_score',
                                 'rca_score', 'total_agatston_score']
        else:
            listFeatureCommon = ['study_site', 'age', 'sex',
                                 'hypertension_nonimpute', 'diabetes_nonimpute',
                                 'hyperlipidemia_nonimpute',
                                 'smokegroup_recalc', 'race_recat',
                                 'numberofvessels', 'lm_score', 'lad_score',
                                 'rca_score', 'total_agatston_score']
    # Build features, append more feature sets as needed
    featureNames = listFeatureCommon

    # Identify datatypes of features
    # featureNameCategorical = ['smokegroup_recalc','race_recat']
    featureNameContinuous = ['age', 'smokegroup_recalc', 'race_recat',
                             'total_score', 'lm_score', 'lad_score',
                             'rca_score', 'numberofvessels', 'total_volscore',
                             'density_recalc_nonimpute', 'mesariskcac']
    if toggleInteraction:
        featureNameContinuous.append('total_score_x_age')    
    featureNameBinary = ['study_site', 'sex', 'hypertension_nonimpute',
                         'diabetes_nonimpute', 'hyperlipidemia_nonimpute',
                         'familyhistory_nonimpute']

    train_data, train_label = \
        get_np(train_data_file, featureNames, out_var)
    test_data, test_label  = \
        get_np(test_data_file, featureNames, out_var)

    if 'study_site' in featureNames:
        # Convert study_site from 1 and 4 to binary 0 and 1
        train_data[:, featureNames.index('study_site')]\
            [train_data[:, featureNames.index('study_site')] == 1] = 0
        train_data[:, featureNames.index('study_site')]\
            [train_data[:, featureNames.index('study_site')] == 4] = 1
        test_data[ :, featureNames.index('study_site')]\
            [test_data[ :, featureNames.index('study_site')] == 1] = 0
        test_data[ :, featureNames.index('study_site')]\
            [test_data[ :, featureNames.index('study_site')] == 4] = 1

    # Compute mean and stdev of continuous variables
    featuresToNorm = [i for i in featureNames if i not in \
                      featureNameBinary]
    meanCont = {}
    stdCont  = {}
    for f in featuresToNorm:
        meanCont[f] = np.nanmean(train_data[:, featureNames.index(f)])
        stdCont[f]  = np.nanstd( train_data[:, featureNames.index(f)])

    # Normalize
    train_data_norm = norm(train_data, featureNames, meanCont, stdCont)
    test_data_norm  = norm(test_data, featureNames, meanCont, stdCont)

    # Check if normalization worked
    if togglePrintNormalization:
        for f in featureNames:
            meanContOne = \
                np.nanmean(train_data[:, featureNames.index(f)])
            meanContNormOne = \
                np.nanmean(train_data_norm[:, featureNames.index(f)])
            stdContOne = \
                np.nanstd(train_data[:, featureNames.index(f)])
            stdContNormOne = \
                np.nanstd(train_data_norm[:, featureNames.index(f)])
            minContOne = \
                np.nanmin(train_data[:, featureNames.index(f)])
            minContNormOne = \
                np.nanmin(train_data_norm[:, featureNames.index(f)])
            maxContOne = \
                np.nanmax(train_data[:, featureNames.index(f)])
            maxContNormOne = \
                np.nanmax(train_data_norm[:, featureNames.index(f)])
            toPrint = ('1:{:10} {:25} Mean:{:6.1f}->{:4.1f}, ' + \
                       'Std:{:6.1f}->{:4.1f}, Range:[{:5.1f},' + \
                       '{:6.1f}]->[{:5.1f},{:5.1f}]')
            print(toPrint.format(
                        modelName, f,
                        meanContOne, meanContNormOne, stdContOne,
                        stdContNormOne, minContOne, maxContOne,
                        minContNormOne, maxContNormOne))

    return (modelName, modelPostFilename, featureNames,
            train_data_norm, test_data_norm, train_data,
            test_data, train_label, test_label, meanCont,
            stdCont)

# Purpose: Prepare data for the model (logistic).
# Arguments: train_data_file: A DataFrame, the training data for the model.
#            test_data_file: A DataFrame, the testing data for the model.
#            out_var: A string, the output variable name.
#            togglePrintNormalization: An int, one to print the variable
#                                      normalizations and zero to not print the
#                                      normalizations. Defaults to zero.
#            toggleInteraction: An int, one to include the interaction term
#                               between age and total score, and zero to
#                               exclude the interaction term. Defaults to one.
# Returns: A tuple containing eleven items:
#          - A string, the model name.
#          - A string, the pickle file name.
#          - A list of strings, containing the feature names.
#          - A NumPy 2-D array, containing the normalized input training
#            data for the model.
#          - A NumPy 2-D array, containing the normalized input testing
#            data for the model.
#          - A NumPy 2-D array, containing the unnormalized input training
#            data for the model.
#          - A NumPy 2-D array, containing the unnormalized input testing
#            data for the model.
#          - A NumPy 1-D array of ints, the output training labels for
#            the model.
#          - A NumPy 1-D array of ints, the output testing labels for
#            the model.
#          - A dictionary, which contains the model's continuous variables as
#            the keys and the variables' corresponding means as the values.
#          - A dictionary, which contains the model's continuous variables as
#            the keys and the variables' corresponding standard deviations as
#            the values.
def prep_mod_data_logistic(train_data_file, test_data_file, out_var,
                           togglePrintNormalization = 0, toggleInteraction = 1):
    modelName = 'Age+Score'
    modelPostFilename = modelName.translate(str.maketrans('', '',
                                            string.punctuation+' '))

    toggleAgeASOnly = 1
    if toggleAgeASOnly:
        if toggleInteraction:
            listFeatureCommon = ['age', 'total_score', 'total_score_x_age']
        else:
            listFeatureCommon = ['age', 'total_score']
    else:
        # FIX!!! - check if smokegroup (0,1,2,3), race_recat (1,2,3,4,5,6)
        # needs to be changed to categorical
        toggleIncludeFamilyHistory = 1
        if toggleIncludeFamilyHistory:
            listFeatureCommon = ['study_site', 'age', 'sex',
                                 'hypertension_nonimpute', 'diabetes_nonimpute',
                                 'familyhistory_nonimpute',
                                 'hyperlipidemia_nonimpute',
                                 'smokegroup_recalc', 'race_recat',
                                 'numberofvessels', 'lm_score', 'lad_score',
                                 'rca_score', 'total_agatston_score']
        else:
            listFeatureCommon = ['study_site', 'age', 'sex',
                                 'hypertension_nonimpute', 'diabetes_nonimpute',
                                 'hyperlipidemia_nonimpute',
                                 'smokegroup_recalc', 'race_recat',
                                 'numberofvessels', 'lm_score', 'lad_score',
                                 'rca_score', 'total_agatston_score']
    # Build features, append more feature sets as needed
    featureNames = listFeatureCommon

    # Identify datatypes of features
    # featureNameCategorical = ['smokegroup_recalc','race_recat']
    featureNameContinuous = ['age', 'smokegroup_recalc', 'race_recat',
                             'total_score', 'lm_score', 'lad_score',
                             'rca_score', 'numberofvessels', 'total_volscore',
                             'density_recalc_nonimpute', 'mesariskcac']
    if toggleInteraction:
        featureNameContinuous.append('total_score_x_age')
    featureNameBinary = ['study_site', 'sex', 'hypertension_nonimpute',
                         'diabetes_nonimpute', 'hyperlipidemia_nonimpute',
                         'familyhistory_nonimpute']

    train_data, train_label = \
        get_np(train_data_file, featureNames, out_var)
    test_data, test_label = \
        get_np(test_data_file, featureNames, out_var)

    if 'study_site' in featureNames:
        # Convert study_site from 1 and 4 to binary 0 and 1
        train_data[:, featureNames.index('study_site')]\
            [train_data[:, featureNames.index('study_site')] == 1] = 0
        train_data[:, featureNames.index('study_site')]\
            [train_data[:, featureNames.index('study_site')] == 4] = 1
        test_data[ :, featureNames.index('study_site')]\
            [test_data[ :, featureNames.index('study_site')] == 1] = 0
        test_data[ :, featureNames.index('study_site')]\
            [test_data[ :, featureNames.index('study_site')] == 4] = 1

    # Compute mean and stdev of continuous variables
    featuresToNorm = [i for i in featureNames if i not in \
                      featureNameBinary]
    meanCont = {}
    stdCont  = {}
    for f in featuresToNorm:
        meanCont[f] = np.nanmean(train_data[:, featureNames.index(f)])
        stdCont[f]  = np.nanstd( train_data[:, featureNames.index(f)])

    # Normalize
    if 0: # Turn off for logistics regression
        train_data_norm = norm(train_data, featureNames, meanCont,
                               stdCont)
        test_data_norm  = norm(test_data, featureNames, meanCont,
                               stdCont)
    else:
        train_data_norm = train_data
        test_data_norm  = test_data

    # Check if normalization worked
    if togglePrintNormalization:
        for f in featureNames:
            meanContOne = \
                np.nanmean(train_data[:, featureNames.index(f)])
            meanContNormOne = \
                np.nanmean(train_data_norm[:, featureNames.index(f)])
            stdContOne = \
                np.nanstd(train_data[:, featureNames.index(f)])
            stdContNormOne = \
                np.nanstd(train_data_norm[:, featureNames.index(f)])
            minContOne = \
                np.nanmin(train_data[:, featureNames.index(f)])
            minContNormOne = \
                np.nanmin(train_data_norm[:, featureNames.index(f)])
            maxContOne = \
                np.nanmax(train_data[:, featureNames.index(f)])
            maxContNormOne = \
                np.nanmax(train_data_norm[:, featureNames.index(f)])
            toPrint = ('1:{:10} {:25} Mean:{:6.1f}->{:4.1f}, ' + \
                       'Std:{:6.1f}->{:4.1f}, Range:[{:5.1f},' + \
                       '{:6.1f}]->[{:5.1f},{:5.1f}]')
            print(toPrint.format(
                        modelName, f,
                        meanContOne, meanContNormOne, stdContOne,
                        stdContNormOne, minContOne, maxContOne,
                        minContNormOne, maxContNormOne))

    return (modelName, modelPostFilename, featureNames,
            train_data_norm, test_data_norm, train_data,
            test_data, train_label, test_label, meanCont,
            stdCont)

# Purpose: Set up the model. For the model, initialize and tune parameters
#          with cross validation (xgboost).
# Arguments: featureNames: A list of strings, containing the model feature
#                          names.
#            numSplits: An int, the number of splits the training data is
#                       partitioned into.
#            dataTrain: A NumPy 2-D array containing the input training data
#                       for the model.
#            labelTrain: A NumPy 1-D array of ints, the output training labels
#                        for the model.
#            seedRandom: An int, the random seed for reproducibility.
# Returns: An xgboost model with tuned parameters.
def setup_classifier_training(featureNames, numSplits, dataTrain,
                              labelTrain, seedRandom):
    # Set seed if not none
    if seedRandom is not None:
        np.random.seed(seedRandom)

    if 0:
        xgb_param = [{'max_depth': np.arange(1, 9, 1),
                      'min_child_weight': np.arange(1, 9, 1)},
                     {'gamma': np.arange(0.1, 0.2, 0.02)},
                     {'subsample': np.arange(0.1, 0.9, 0.1),
                      'colsample_bytree': np.arange(0.1, 0.9, 0.1)},
                     {'n_estimators': np.arange(100, 301, 100)},
                     {'reg_alpha': [1e-5, 1e-2, 1]}]
    else:
        xgb_param={ 'objective': ['binary:logistic'],
                    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    'n_estimators': np.arange(50, 250, 100),
                    'max_depth': np.arange(1, 9, 2),
                    'min_child_weight': np.arange(1, 9, 2),
                    'subsample': np.arange(0.1, 0.5, 0.2),
                    'gamma': np.arange(0, 0.6, 0.2),
                    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10],
                    'colsample_bytree': np.arange(0.1, 0.9, 0.2)}
        
    toggleTempFastParamsForDebugging = 1
    if toggleTempFastParamsForDebugging:
        # TEMP for fast debuggin (turn off after done)
        xgb_param = [{'max_depth': np.arange(1, 9, 1)}]

    # Do cross validation split
    indexTrainTestCV = []
    skf = StratifiedKFold(n_splits = 5)
    for indexTrain in skf.split(dataTrain, labelTrain):
        indexTrainTestCV.append(indexTrain)
    # Setup CV hyperparameter search
    xgb = GridSearchCV(estimator = XGBClassifier(max_iter = 250),
                       param_grid = xgb_param, scoring = 'roc_auc',
                       cv = indexTrainTestCV)  # iid=True
                                               # removed since to be deprecated
    modelGSCV = xgb
    return modelGSCV

# Purpose: Set up the model. For the model, initialize and tune parameters
#          with cross validation (logistic).
# Arguments: featureNames: A list of strings, containing the model feature
#                          names.
#            numSplits: An int, the number of splits the training data is
#                       partitioned into.
#            dataTrain: A NumPy 2-D array containing the input training data
#                       for the model.
#            labelTrain: A NumPy 1-D array of ints, the output training labels
#                        for the model.
#            seedRandom: An int, the random seed for reproducibility.
# Returns: A logistic regression model with tuned parameters.
def setup_classifier_training_logistic(featureNames, numSplits,
                                       dataTrain, labelTrain,
                                       seedRandom):
    # Set seed if not none
    if seedRandom is not None:
        np.random.seed(seedRandom)
        
    if 0:
        lgr_param = {'C': np.logspace(-6, 3, 10),
                     'penalty': ['l1', 'l2', 'none'],
                     'solver': ['liblinear', 'saga']}
    else:
        lgr_param = {'penalty': ['none']}
    
    toggleTempFastParamsForDebugging = 1
    if toggleTempFastParamsForDebugging:
        # TEMP for fast debuggin (turn off after done)
        xgb_param = [{'max_depth': np.arange(1, 9, 1)}]

    # Do cross validation split
    indexTrainTestCV = []
    skf = StratifiedKFold(n_splits = 5)
    for indexTrain in skf.split(dataTrain, labelTrain):
        indexTrainTestCV.append(indexTrain)
    # Setup CV hyperparameter search
    lgr = GridSearchCV(estimator = LogisticRegression(max_iter = 250),
                       param_grid = lgr_param, scoring = 'roc_auc',
                       cv = indexTrainTestCV) # iid=True
                                              # removed since to be deprecated
    modelGSCV = lgr
    return modelGSCV

# Purpose: Fits the model, with its tuned parameters, to the training data.
# Arguments: modelName: A string, the model name.
#            modelPostFilename: A string, the pickle file name.
#            numSplits: An int, the number of splits the training data is
#                       partitioned into.
#            dataTrain: A NumPy 2-D array containing the input training
#                       data for the model.
#            labelTrain: A NumPy 1-D array of ints, the output training
#                        labels for the model.
#            modelGSCV: A model with tuned parameters.
#            togglePrintTiming: An int, if 1, time the fitting for the model.
#                               If 0, no timing is performed.
# Returns: A fitted model.
def fit_mod(modelName, modelPostFilename, numSplits, dataTrain,
            labelTrain, modelGSCV, togglePrintTiming):
    togglePerformTraining  = 1
    toggleSaveTrainedModel = 1

    if not togglePerformTraining:
        modelGSCV = None

    strSaveModelPKL = dirnamePKL + '\\' + 'SavedModel_' + \
                      modelPostFilename + '_00' + '.pkl'

    if togglePerformTraining:
        if togglePrintTiming:
            print('Fitting Model:', modelName[:15], ', Split: 0', end = '')
        start = timer()

        # Fit the model to the training data
        modelGSCV.fit(dataTrain, labelTrain)

        end = timer()
        if togglePrintTiming:
            print(', Time(s): {:.1f}'.format(end - start),
                  end = '') # Time in seconds

        if toggleSaveTrainedModel:
            pickle.dump(modelGSCV, open(strSaveModelPKL, 'wb'))
            if togglePrintTiming:
                print(', Saved as', strSaveModelPKL[:25])
    else:
        xgb = pickle.load(open(strSaveModelPKL, 'rb'))
        modelGSCV = xgb

    return modelGSCV

# Purpose: Calculates performance metrics for the classifier.
# Arguments: TP: An int, the number of positives the model predicts correctly.
#            FP: An int, the number of positives the model predicts incorrectly.
#            FN: An int, the number of negatives the model predicts incorrectly.
#            TN: An int, the number of negatives the model predicts correctly.
#            roc_auc: A float, the roc auc score.
# Returns: A tuple of twelve values: 
#          - An int, the total number of actual positive cases.
#          - An int, the total number of actual negative cases.
#          - An int, the total number of predicted positive cases.
#          - An int, the total number of predicted negative cases.
#          - An int, the total number of cases.
#          - A float, the Matthews correlation coefficient.
#          - A float, the sensitivity of the classifier.
#          - A float, the ppv of the classifier.
#          - A float, the specificity of the classifier.
#          - A float, the npv of the classifier.
#          - A float, the prevalence.
#          - A float, the accuracy of the classifier.
def calcStats(TP, FP, FN, TN, roc_auc):
    # Calculate stats
    tot_act_pos = TP + FN
    tot_act_neg = FP + TN
    tot_pred_pos  = TP + FP
    tot_pred_neg  = FN + TN
    totPop  = TP + FP + FN + TN
    MCC = (TP * TN - FP * FN) / \
          np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) *( TN + FN))
    sensitivity = 100 * TP / tot_act_pos
    ppv = 100 * TP / tot_pred_pos
    specificity = 100 * TN / tot_act_neg
    npv = 100 * TN / tot_pred_neg
    prevalence = 100 * tot_act_pos / totPop
    accuracy = 100 * (TP + TN) / totPop
    return (tot_act_pos, tot_act_neg, tot_pred_pos, tot_pred_neg, totPop, MCC, 
            sensitivity, ppv, specificity, npv, prevalence, accuracy)

# Purpose: Prints the calculated performance metrics of the classifier.
# Arguments: TP: An int, the number of positives the model predicts correctly.
#            FP: An int, the number of positives the model predicts incorrectly.
#            FN: An int, the number of negatives the model predicts incorrectly.
#            TN: An int, the number of negatives the model predicts correctly.
#            roc_auc: A float, the roc auc score.
#            roc_std: A float, the standard error of the auc score.
#            roc_ci: A list of two floats, the lower bound and the upper bound
#                    of the auc score 95% confidence interval.
#            tot_act_pos: An int, the total number of actual positive cases.
#            tot_act_neg: An int, the total number of actual negative cases.
#            totPop: An int, the total number of cases.
#            MCC: A float, the Matthews correlation coefficient.
#            sensitivity: A float, the sensitivity of the classifier.
#            ppv: A float, the ppv of the classifier.
#            specificity: A float, the specificity of the classifier.
#            npv: A float, the npv of the classifier.
#            prevalence: A float, the prevalence.
#            accuracy: A float, the accuracy of the classifier.
# Returns: Nothing.
def printStats(TP, FP, FN, TN, roc_auc, roc_std, roc_ci, tot_act_pos,
               tot_act_neg, totPop, MCC, sensitivity, ppv, specificity, npv,
               prevalence, accuracy):
    print('[ TP, FP ]=[%5d,%5d]' % (TP, FP))
    print('[ FN, TN ]=[%5d,%5d]' % (FN, TN))
    print('Pos  = %5d\t' % (tot_act_pos), end= '')
    print()
    print('Neg  = %5d\t' % (tot_act_neg), end= '')
    print('Sens = %3.0f%%\t' % (sensitivity), end= '')
    print('PPV  = %3.0f%%\t' % (ppv), end= '')
    print()
    print('Pop  = %5d\t' % (totPop), end= '')
    print('Spec = %3.0f%%\t' % (specificity), end= '')
    print('NPV  = %3.0f%%\t' % (npv), end= '')
    print()
    print('Prev = %5.0f%%\t' % (prevalence), end= '')   
    print('Acc  = %3.0f%%\t' % (accuracy), end= '')
    print('AUC = {:.4f} +/- {:.4f} CI(95%):[{:.4f},{:.4f}]'.format( \
                roc_auc, roc_std, roc_ci[0], roc_ci[1] ))
    print('MCC = %.2f\t' % (MCC), end= '')
    print()

# Purpose: Gets the classifier metrics for the model.
# Arguments: modelGSCV: The fitted model.
#            modelName: A string, the model name.
#            numSplits: An int, the number of splits the testing data is
#                       partitioned into.
#            dataTest: A NumPy 2-D array containing the input testing data for
#                      the model.
#            labelTest: A NumPy 1-D array of ints, the output testing labels for
#                       the model.
#            classifierThreshold: A float, the classifier threshold.
# Returns: A tuple of twenty values:
#          - An int, the number of positives the model predicts correctly.
#          - An int, the number of positives the model predicts incorrectly.
#          - An int, the number of negatives the model predicts incorrectly.
#          - An int, the number of negatives the model predicts correctly.
#          - A float, the roc auc score.
#          - A float, the standard error of the auc score.
#          - A list of two floats, the lower bound and the upper bound
#            of the auc score 95% confidence interval.
#          - A list of two floats, the false positive rate and the true
#            positive rate (for the roc curve).
#          - An int, the total number of actual positive cases.
#          - An int, the total number of actual negative cases.
#          - An int, the total number of predicted positive cases.
#          - An int, the total number of predicted negative cases.
#          - An int, the total number of cases.
#          - A float, the Matthews correlation coefficient.
#          - A float, the sensitivity of the classifier.
#          - A float, the ppv of the classifier.
#          - A float, the specificity of the classifier.
#          - A float, the npv of the classifier.
#          - A float, the prevalence.
#          - A float, the accuracy of the classifier.
def get_roc_auc_values(modelGSCV, modelName, numSplits,
                       dataTest, labelTest, classifierThreshold):
    testProb = np.array(modelGSCV.predict_proba(dataTest)[:, 1],
                        dtype = float)
    testPred = np.array(testProb >= classifierThreshold, dtype = int)
    
    roc_auc = roc_auc_score(labelTest, testProb)
    auc, auc_var, auc_ci = auc_ci_Delong(labelTest, testProb)
    roc_std = np.sqrt(auc_var)
    roc_ci = auc_ci
    
    fpr, tpr,_  = roc_curve(labelTest, testProb)
    roc_curve_values = [fpr, tpr]
    
    TP = np.sum(np.logical_and(testPred == 1, labelTest == 1))
    FP = np.sum(np.logical_and(testPred == 1, labelTest == 0))
    FN = np.sum(np.logical_and(testPred == 0, labelTest == 1))
    TN = np.sum(np.logical_and(testPred == 0, labelTest == 0))
    
    (tot_act_pos, tot_act_neg, tot_pred_pos, tot_pred_neg, totPop, MCC, 
             sensitivity, ppv, specificity, npv, prevalence, accuracy) = \
             calcStats(TP, FP, FN, TN, roc_auc)
    
    return (TP, FP, FN, TN, roc_auc, roc_std, roc_ci, roc_curve_values,
            tot_act_pos, tot_act_neg, tot_pred_pos, tot_pred_neg, totPop, MCC, 
            sensitivity, ppv, specificity, npv, prevalence, accuracy)


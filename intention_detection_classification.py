#!/usr/bin/env python
# coding: utf-8

"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script prepare IMU data for terrain classification.
"""

import os
import pandas as pd
import glob


import random
from random import randrange

from sklearn.model_selection import train_test_split  # preprocessing
from sklearn.preprocessing import StandardScaler  # preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # feature selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# import/export
import joblib

# DEFINITIONS
WIN_SIZE = 32  # window size
USER = 'Mahsa'  # determine the study participant
featureSet = 'ALL_TORQUE'  # choose the feature subsets for clustering
n_components = 6

maneuvers = ['Obstacles15', 'Obstacles35', 'RampA', 'StraightF', 'Turn90FR', 'Turn90FL', 'Turn180L', 'Turn180R']
trials = ['T1', 'T2', 'T3']  # trials

# unused columns for correlation analysis
unused_cols = ['maneuver', 'trial', 'Prob_L0', 'Prob_L1', 'Prob_L2', 'Prob_L3', 'Prob_L4', 'Prob_L5']
Labels = ['0', '1', '2', '3', '4', '5']  # trials

CURR_PATH = os.path.abspath('.')  # path to save labeled data and corresponding figures
glob_paths = glob.glob(os.path.join(CURR_PATH, 'labeled_data', USER, str(WIN_SIZE), '*.csv'))  # path train datasets


def split_train_test(df):

    """Split train/test data"""

    random.seed(1)
    test_trials = []
    test_ = pd.DataFrame(columns=df.columns)
    train_ = pd.DataFrame(columns=df.columns)

    for maneuver in maneuvers:

        if 'Ramp' in maneuver and USER == 'Mahsa':
            test_tr = trials[randrange(2)]
        else:
            test_tr = trials[randrange(3)]

        test_trials.append(maneuver+'_'+test_tr)

        df_TE = labeled_df.loc[labeled_df['trial'] == maneuver + '_' + test_tr + '_WS' + str(WIN_SIZE) + '_' + USER]
        test_ = test_.append(df_TE, ignore_index=True)
        train_tr = [tr for tr in trials if tr != test_tr]

        for tr in train_tr:
            df_TR = labeled_df.loc[labeled_df['trial'] == maneuver + '_' + tr + '_WS' + str(WIN_SIZE) + '_' + USER]
            train_ = train_.append(df_TR, ignore_index=True)

    return train_, test_


def sub_columns(df):
    """ select all torque features"""
    torque = [col for col in df.columns.tolist() if 'Torque' in col]
    return torque


def create_train_test(dataset, target_label, test_size):
    """
    input: get dataset and desired target label to perform classification
    output: train/test splits
    """
    df = dataset.copy()
    df = df.drop(columns=unused_cols)
    y = df.pop(target_label)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=0, shuffle=False)

    return X_train, X_test, y_train, y_test


def clf_pipeline(X_train, y_train):
    """Classification pipeline"""

    cv = 5  # cross validation nfolds

    # classifier to use for feature selection
    feat_selection_clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=0)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('selector', SFS(estimator=feat_selection_clf,
                                      k_features=(2, X_train.shape[1]),
                                      forward=True,
                                      floating=False,
                                      scoring='accuracy',
                                      cv=cv,
                                      n_jobs=-1)),
                     ('classifier', RandomForestClassifier())])

    param_grid = [
        {'selector': [SFS(estimator=feat_selection_clf)],
         'selector__estimator__n_estimators':[50],
         'selector__estimator__max_depth':[10]},

        {'classifier':[RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)],
         'classifier__n_estimators':[50],
         'classifier__max_depth':[10]}]

    scoring = {
        'precision_score': make_scorer(precision_score, average='macro'),
        'recall_score': make_scorer(recall_score, average='macro'),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average='macro')}

    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=scoring,
                        n_jobs=-1,
                        refit='accuracy_score',
                        cv=cv,
                        verbose=0,
                        return_train_score=True)

    grid.fit(X_train, y_train)

    return grid


labeled_df = pd.read_csv(glob_paths[0])  # Read data and update current user dictionary
labeled_df['Clus_label'] = labeled_df['Clus_label'].astype(float)
torque_columns = sub_columns(labeled_df)
train_df, test_df = split_train_test(labeled_df)
X_train, X_, y_train, y_ = create_train_test(train_df, 'Clus_label', test_size=1)
X_, X_test, y_, y_test = create_train_test(test_df, 'Clus_label', test_size = len(test_df)-1)
X_train = X_train[torque_columns]
X_test = X_test[torque_columns]

# run the classification pipeline
model_ = clf_pipeline(X_train, y_train)

# get parameters of the best model
print("Best estimator via GridSearch \n", model_.best_estimator_)

# Export Model
model_path = os.path.join(CURR_PATH, 'classification_model')  # create directory to save results
os.makedirs(model_path)
model_name = os.path.join(model_path, 'rf.joblib')
joblib.dump(model_, model_name)  # dump model

print("SUCCESSFULLY EXECUTED!!!!")

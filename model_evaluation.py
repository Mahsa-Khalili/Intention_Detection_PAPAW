#!/usr/bin/env python
# coding: utf-8

"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script prepare IMU data for terrain classification.
"""

import os
import pandas as pd
import numpy as np
import glob
import pathlib


import random
from random import randrange

# plotting
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
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

# create a color pallette
cmap = matplotlib.cm.get_cmap('tab10')
color_ = [cmap(i) for i in range(int(n_components))]

CURR_PATH = os.path.abspath('.')  # path to save labeled data and corresponding figures
glob_paths = glob.glob(os.path.join(CURR_PATH, 'labeled_data', '*.csv'))  # path train datasets
dataset_path = [path for path in glob_paths if 'gmm' in path]

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


def plt_confusion_matrix(clmodel, xtest, ytest):
    # confusion matrix
    title = "Normalized confusion matrix"

    disp = plot_confusion_matrix(clmodel, xtest, ytest, display_labels=Labels,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation=45,
                                 values_format='.2f')
    disp.ax_.set_title(title)
    disp.ax_.grid(False)
    disp.figure_.set_size_inches(12, 12)
    plt.show()


def sfs_eval(model_, xtest):
    # get best features
    best_feats_idx = model_.best_estimator_['selector'].k_feature_idx_
    best_feats = list(xtest.columns[list(best_feats_idx)].values.tolist())
    print('\nBest features: \n{}'.format(best_feats))

    # plotting feature selection characteristics
    plot_sfs(model_.best_estimator_['selector'].get_metric_dict(), kind='std_err', figsize=(12, 5))
    plt.title('Sequential Forward Selection (w. StdErr)')
    plt.grid(b=True, which='major', axis='both')
    plt.show()


def range_dic_(df_):
    """
    get the start index of each maneuver from the original dataframe
    """
    range_dic = {}
    for man in df_['maneuver']:
        trial_indx = df_.index[df_['maneuver'] == man].tolist()
        range_ = (min(trial_indx), max(trial_indx))
        range_dic.update({man: range_})
    return range_dic


def plt_ts_cluster_prediction(df_clus_, predictions, clusterNum, features_to_plot, man_type='All', zoom=False):
    """
    input: input original dataframe (with maneuver columns), clustered dataframe, number of clusteres,
           and selected features to plot
    output: plotting clustered time series data with different colors
    """

    MARKER_SIZE = 5
    LINE_WIDTH = 2
    plt_num = len(features_to_plot)
    color_dict = {}

    man_range = range_dic_(test_df)

    fig, axs = plt.subplots(plt_num, 1, figsize=(15, 15), constrained_layout=True)
    axs = axs.ravel()

    if man_type != 'All':
        df_clus = df_clus_[man_range[man_type][0]:man_range[man_type][1]]

    true_states = df_clus['Clus_label'].astype(int)
    predicted_states = predictions[man_range[man_type][0]:man_range[man_type][1]].astype(int)

    colors = [cmap(i) for i in range(clusterNum)]

    for i in range(clusterNum):
        color_dict.update({i: colors[i]})

    color_array_true = [color_dict[i] for i in true_states]
    color_array_predicted = [color_dict[i] for i in predicted_states]

    for i, feature in enumerate(features_to_plot):
        axs[i].grid()
        axs[i].scatter(range(len(df_clus)), df_clus[feature], facecolors='none', edgecolors=color_array_true,
                       linewidth=2*LINE_WIDTH, s=5*MARKER_SIZE)
        axs[i].scatter(range(len(df_clus)), df_clus[feature], c=color_array_predicted, s=5*MARKER_SIZE, marker='o')
        axs[i].set_ylabel(feature + ' (Nm)', fontsize=15)
        axs[i].tick_params(direction='out', labelsize=15)
        axs[i].set_xlim((0, len(df_clus)))

        if zoom:
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            # create the zoomed in version
            axins = zoomed_inset_axes(axs[i], 3, loc=1)  # zoom-factor: 3, location: upper-left
            axins.scatter(range(len(df_clus)), df_clus[feature], facecolors='none', edgecolors=color_array_true,
                          linewidth=5, s=20*MARKER_SIZE)
            axins.scatter(range(len(df_clus)), df_clus[feature], c=color_array_predicted, s=10*MARKER_SIZE,
                          marker='o')
            x1, x2, y1, y2 = 25, 55, -1, 1  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # dummy scatter for labels
        axs[i].scatter([], [], edgecolors=colors[0], label='Recovery (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)
        axs[i].scatter([], [], edgecolors=colors[1], label='Right-assist (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)
        axs[i].scatter([], [], edgecolors=colors[2], label='Straight-assist (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)
        axs[i].scatter([], [], edgecolors=colors[3], label='Release (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)
        axs[i].scatter([], [], edgecolors=colors[4], label='Braking (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)
        axs[i].scatter([], [], edgecolors=colors[5], label='Left-assist (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s=20*MARKER_SIZE)

        axs[i].scatter([], [], c=np.array(colors[0]).reshape(1, -1), marker='o', label='Recovery (Predicted)', s=10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[1]).reshape(1, -1), marker='o', label='Right-assist (Predicted)', s=10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[2]).reshape(1, -1), marker='o', label='Straight-assist (Predicted)', s=10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[3]).reshape(1, -1), marker='o', label='Release (Predicted)', s=10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[4]).reshape(1, -1), marker='o', label='Braking (Predicted)', s=10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[5]).reshape(1, -1), marker='o', label='Left-assist (Predicted)', s=10*MARKER_SIZE)

    axs[i].legend(ncol=2, fontsize=10)

    plt.show()


labeled_df = pd.read_csv(dataset_path[0])  # Read data and update current user dictionary
labeled_df['Clus_label'] = labeled_df['Clus_label'].astype(float)
torque_columns = sub_columns(labeled_df)
train_df, test_df = split_train_test(labeled_df)
X_train, X_, y_train, y_ = create_train_test(train_df, 'Clus_label', test_size=1)
X_, X_test, y_, y_test = create_train_test(test_df, 'Clus_label', test_size=len(test_df)-1)
X_train = X_train[torque_columns]
X_test = X_test[torque_columns]

# import classification model
model_path = os.path.join(CURR_PATH, 'classification_model')  # create directory to save results
model_name = os.path.join(model_path, 'rf.joblib')
model_ = joblib.load(model_name)

# accuracy
print('Accuracy score = {:0.2f}'.format(model_.score(X_test, y_test) * 100, '.2f'))
print("Best parameters via GridSearch \n", model_.best_params_)
print("Best estimator via GridSearch \n", model_.best_estimator_)  # get parameters of the best model

y_pred = model_.predict(X_test)
f_score = f1_score(y_test, y_pred, average='macro') * 100
recall_ = recall_score(y_test, y_pred, average='macro') * 100
print('F1-score = {:0.2f}, Recall score = {:0.2f}'.format(f_score, recall_))

# plt_confusion_matrix(model_, X_test, y_test)
sfs_eval(model_, X_test)
#
# features_to_plot = ['Mean Torque_L', 'Mean Torque_R']
# plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type='StraightF')
#
# features_to_plot = ['Mean Torque_L', 'Mean Torque_R']
# plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type='Turn90FR')
#
# features_to_plot = ['Mean Torque_L', 'Mean Torque_R']
# plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type='Turn90FL')

# if SAVE_DATA:
X_test.insert(0, 'rf_label', y_pred)
processed_path = os.path.join(CURR_PATH, 'labeled_data')
pathlib.Path(processed_path).mkdir(parents=True, exist_ok=True)
filename = 'rf_labels.csv'
filename = os.path.join(processed_path, filename)
X_test.to_csv(filename, index=False)

print("SUCCESSFULLY EXECUTED!!!!")

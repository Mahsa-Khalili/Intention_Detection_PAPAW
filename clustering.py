#!/usr/bin/env python
# coding: utf-8

"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script prepare IMU data for terrain classification.
"""

import os
import glob
import pathlib

import random
import numpy as np
import pandas as pd

import pickle
import joblib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import itertools
from scipy import linalg
import matplotlib as mpl
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn import mixture


# DEFINITIONS
USER = 'Mahsa'  # ['Mahsa', 'Jaimie']  # participant name
WIN_SIZE = 32  # window size

# list of all maneuvers
maneuvers = ['Obstacles15', 'Obstacles35', 'RampA', 'StraightF', 'Turn90FR', 'Turn90FL', 'Turn180L', 'Turn180R']

# choose the feature subsets for clustering
featureSet_list = ['ALL', 'ALL_TORQUE', '2D_TORQUE', 'LR_TORQUE', 'LR_TORQUE_MEAN', '2D_TORQUE_MEAN'] 

dataset_to_import = 'featured_data'  # choose dataset/datasets to import

featured_columns = ['AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel', 'Torque_L', 'Torque_R',
                    'Torque_sum', 'Torque_diff', 'Torque_L_roc', 'Torque_R_roc']

time_features = ['Mean', 'Std', 'Max', 'Min', 'RMS']

# clustering model parameters
clus_params = {'covar_types': 'full', 'n_components': 6, 'feat_list': 'ALL_TORQUE'}

# path to save labeled data and corresponding figures
CURR_PATH = os.path.abspath('.')

# Import processed data
dataset_paths = glob.glob(os.path.join(CURR_PATH, dataset_to_import, USER, 'WinSize' + str(WIN_SIZE), '*.csv'))

# create a color pallette
cmap = matplotlib.cm.get_cmap('tab10')


def import_func(path_):
    """ function to import featured datasets"""

    datasets_dic = {}

    for dataset_path in path_:
        # Parse labels from filenames
        dataset_label = os.path.split(dataset_path)[1].split('.')[0]

        # Read from csv to Pandas
        dataset = pd.read_csv(dataset_path)

        # insert dataset label to the dataframes
        dataset.insert(0, 'trial', dataset_label)
        dataset.insert(0, 'maneuver', dataset_label.split('_')[0])

        # Datasets are stored in a dictionary
        datasets_dic.update({dataset_label: dataset})

    # list of imported maneuvers
    dataset_names = list(datasets_dic.keys())

    return datasets_dic, dataset_names


def prep_func(data_dic):
    """Prepare dataframes for clustering"""

    df_all = pd.DataFrame(columns=datasets[dataset_labels[0]].columns.tolist())

    # combine desired datasets into one dataframe
    for label in dataset_labels:
        df_all = pd.concat([df_all, data_dic[label]], ignore_index=True)

    df_all_columns = df_all.copy()  # keep a copy of the original dataframes before dropping the trial names

    # dropping unused columns/features
    for col in ['Time', 'trial', 'maneuver']:
        if col in df_all.columns:
            df_all = df_all.drop(columns=[col])

    columns_all = df_all.columns.tolist()
    columns_torque = [col for col in df_all.columns.tolist() if 'Torque' in col]  # all torque data

    # all torque features except for roc (mean/std/...  & left/right/sum/diff)
    columns_2d_torque = [col for col in df_all.columns.tolist()
                         if 'Torque_sum' in col or 'Torque_diff' in col and 'roc' not in col]

    # all torque features of left and right only (mean/std/...  & left/right)
    columns_lr_torque = [col for col in df_all.columns.tolist()
                         if ('Torque_L' in col or 'Torque_R' in col) and 'roc' not in col]

    columns_lr_torque_mean = ['Mean Torque_L', 'Mean Torque_R']  # mean torque left and right only
    columns_2d_torque_mean = ['Mean Torque_sum', 'Mean Torque_diff']  # mean torque left and right only

    # dictionary of list of feature subsets to be used for dimension_reduction or clustering
    featureSet_dic = {'ALL': columns_all, 'ALL_TORQUE': columns_torque,
                      '2D_TORQUE': columns_2d_torque, '2D_TORQUE_MEAN': columns_2d_torque_mean,
                      'LR_TORQUE': columns_lr_torque, 'LR_TORQUE_MEAN': columns_lr_torque_mean}

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    feat_all_stand = scaler.fit_transform(df_all.values)
    df_all_stand = pd.DataFrame(feat_all_stand, columns=data_columns)  # normalized dataset

    return df_all_stand, df_all_columns, featureSet_dic


def clus_func(df_all, n_components, feat_subset):
    """
    function to cluster and evaluate the clustering performance
    input: dataframe consisting of different maneuvers to be clustered, feature sets to be used for clustering,
           and the clustering model
    output: labeled dataframe and three performance measures
    """

    df = df_all[featureSet_dic[feat_subset]].copy()

    X = df.values

    # # Fit a Gaussian mixture with EM
    # gmm_model = mixture.GaussianMixture(n_components=n_components,
    #                                     covariance_type=cv_type,
    #                                     random_state=1,
    #                                     n_init=10)
    # gmm_model = gmm_model.fit(X)

    model_path = os.path.join(CURR_PATH, 'clustering_model')  # create directiry for the current time
    model_name = os.path.join(model_path, 'gmm.joblib')
    gmm_model = joblib.load(model_name)

    # predic labels & probabilities
    labels = gmm_model.predict(X)
    labels_prob = gmm_model.predict_proba(X)

    # adding all droped features (for plotting purposes) of the standardized dataframe
    added_feat = [feat for feat in data_columns if feat not in df.columns]
    df[added_feat] = df_all_stand[added_feat].copy()
    df = df[data_columns]

    # adding the labels to the dataframe
    df.insert(0, 'Clus_label', labels)

    for n in range(n_components):
        df['Prob_L'+str(n)] = labels_prob[:, n]

    return gmm_model, df  # export all gmm models and a dictionary of all labeled datasets


def labeling_func(df_clus):
    """ add all cluster labels to the original dataframe """

    df_all_labeled = df_all_columns.copy()
    df_all_labeled['Clus_label'] = df_clus['Clus_label'].copy()
    df_all_labeled['Clus_label']= df_all_labeled['Clus_label'].astype(int)
    for i in range(0, clus_params['n_components']):
        df_all_labeled['Prob_L'+str(i)] = df_clus['Prob_L'+str(i)].copy()

    return df_all_labeled


def plt_gm_clusters(df_all, model):

    """this function gets unlabeled scaled dataframe and predict labels + plotting cluster ellips"""

    # color_iter = itertools.cycle([cmap(i) for i in range(cmap.N)])

    color_iter = itertools.cycle([cmap(i) for i in range(clus_params['n_components'])])

    df = df_all[featureSet_dic[clus_params['feat_list']]].copy()

    XX = df.values
    Y_ = model.predict(XX)  # predict labels for each model

    plt.figure(figsize=(8, 6))
    splot = plt.subplot(1, 1, 1)

    for i, (mean, cov, color) in enumerate(zip(model.means_, model.covariances_, color_iter)):

        if "MEAN" in clus_params['feat_list']:
            v, w = linalg.eigh(cov)
        else:

            subset = [0, 5]  # mean torque L & R
            v, w = linalg.eigh(cov[np.ix_(subset, subset)])
            mean = np.array([mean[0], mean[5]])

        if not np.any(Y_ == i):
            continue

        if "MEAN" in clus_params['feat_list']:
            plt.scatter(XX[Y_ == i, 0], XX[Y_ == i, 1], color=color, s=60)
        else:
            plt.scatter(XX[Y_ == i, 0], XX[Y_ == i, 5], color=color, s=60)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())

    plt.title('Subject: {}, feature set: {}'.format(USER, clus_params['feat_list']))
    plt.subplots_adjust(hspace=.35, bottom=.02)
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


def plt_ts_cluster(df_, features_to_plot):
    """
    input: input original dataframe (with maneuver columns), clustered dataframe, number of clusteres,
           and selected features to plot
    output: plotting clustered time series data with different colors
    """

    df_clus = df_.copy()

    plt_num = 2

    fig, axs = plt.subplots(plt_num, 1, figsize=(15, 12), constrained_layout=True)
    axs = axs.ravel()

    states = df_clus['Clus_label']

    clusterNum = clus_params['n_components']

    color_dict = {i:cmap(i) for i in range(clusterNum)}
    color_array = [color_dict[i] for i in states]

    for i, feature in enumerate(features_to_plot):
        axs[i].scatter(df_clus.index, df_clus[feature], c=color_array, s=10)
        axs[i].set_xlim([-1, len(df_clus)+1])
        axs[i].tick_params(direction='out', labelsize=15)
        axs[i].yaxis.grid(True)

        if 'Torque' in feature:
            axs[i].set_ylabel(feature + ' (Nm)', fontsize=15)
        elif 'Lin' in feature:
            axs[i].set_ylabel(feature + ' (m/s)', fontsize=15)
        elif 'Ang' in feature:
            axs[i].set_ylabel(feature + ' (rad/s)', fontsize=15)

    fig.suptitle(clus_params['feat_list'], fontsize=16)

    range_dic = range_dic_(df_clus)

    for trial, range_ in range_dic.items():
        axs[0].text(range_[0], axs[0].get_ylim()[1]+0.2, trial, fontsize=15, rotation=45)
        for i in range(plt_num):
            axs[i].axvline(x=range_[0], linestyle='--', linewidth=0.5)

    plt.show()


# function to plot clusters in time series data
def plt_ts_cluster_subset(df_, features_to_plot, man_list=maneuvers):
    """
    input: input original dataframe (with maneuver columns), clustered dataframe, number of clusteres,
           and selected features to plot
    output: plotting clustered time series data with different colors
    """

    clusterNum = clus_params['n_components']
    color_dict = {i: cmap(i) for i in range(clusterNum)}
    figsize = (15, 15)
    legend_size = 15

    if len(man_list) == 1:
        figsize = (15, 8)

    fig, axs = plt.subplots(len(man_list), 1, figsize=figsize, constrained_layout=True)
    fig.suptitle(clus_params['feat_list'], fontsize=16)

    if len(man_list) != 1:
        axs = axs.ravel()

    for i, wheelchair_man in enumerate(man_list):

        df_clus = df_.loc[df_['maneuver'] == wheelchair_man].copy()
        df_clus = df_clus.reset_index()

        states = df_clus['Clus_label']
        color_array = [color_dict[i] for i in states]

        if len(man_list) != 1:

            axs[i].scatter(df_clus.index, df_clus[features_to_plot[0]], c=color_array, s=16)
            axs[i].scatter(df_clus.index, df_clus[features_to_plot[1]], c=color_array, s=16, alpha=0.7, marker='>')

            axs[i].tick_params(direction='out', labelsize=15)
            axs[i].set_ylabel('Torque (Nm)', fontsize=15)
            axs[i].set_title(wheelchair_man)
            axs[i].yaxis.grid(True)
            axs[i].set_xlim([-1, len(df_clus)+1])

            legend_elements = [Line2D([0], [0], marker='>', color='w', label='Right',
                                      markerfacecolor='k', markersize=15),
                               Line2D([0], [0], marker='o', color='w', label='Left',
                                      markerfacecolor='k', markersize=15)]
            axs[i].legend(handles=legend_elements, fontsize=legend_size)

        else:
            axs.scatter(df_clus.index, df_clus[features_to_plot[0]], c=color_array, s=16)
            axs.scatter(df_clus.index, df_clus[features_to_plot[1]], c=color_array, s=16, alpha=0.7, marker='>')

            axs.tick_params(direction='out', labelsize=15)
            axs.set_ylabel('Torque (Nm)', fontsize=15)
            axs.set_title(wheelchair_man)
            axs.yaxis.grid(True)
            axs.set_xlim([-1, len(df_clus)+1])

            legend_elements = [Line2D([0], [0], marker='>', color='w', label='Right',
                                      markerfacecolor='k', markersize=15),
                               Line2D([0], [0], marker='o', color='w', label='Left',
                                      markerfacecolor='k', markersize=15)]
            axs.legend(handles=legend_elements, fontsize=legend_size)

    plt.show()


datasets, dataset_labels = import_func(dataset_paths)
data_columns = [col for col in datasets[dataset_labels[0]].columns if col != 'trial' and
                col != 'Time' and col != 'maneuver']  # get columns/features of the imported datasets
df_all_stand, df_all_columns, featureSet_dic = prep_func(datasets)

# run the cluster function or import trained model
models, df_clus = clus_func(df_all_stand.copy(),
                            clus_params['n_components'],
                            clus_params['feat_list'])

df_labeled = labeling_func(df_clus)  # adding labels to all datastes

plt_gm_clusters(df_all_stand.copy(), models)  # plotting cluster over torque left/right

plt_ts_cluster(df_labeled, ['Mean Torque_L', 'Mean Torque_R'])  # plotting all labeled data in a time series format

# plotting a subset of selected labeled maneuvers in a time series format
plt_ts_cluster_subset(df_labeled, ['Mean Torque_L', 'Mean Torque_R'], ['StraightF'])

# if SAVE_DATA:
processed_path = os.path.join(CURR_PATH, 'labeled_data')
pathlib.Path(processed_path).mkdir(parents=True, exist_ok=True)
filename = "gmm_labels.csv"
filename = os.path.join(processed_path, filename)
df_labeled.to_csv(filename, index=False)

print("SUCCESSFULLY EXECUTED!!!!")

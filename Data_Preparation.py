#!/usr/bin/env python
# coding: utf-8

"""
Author:         Mahsa Khalili
Date:           2021 April 15th
Purpose:        This Python script prepare IMU data for terrain classification.
"""

import os
import pathlib
import glob

import pandas as pd
import numpy as np

from scipy import signal
from scipy.signal import sosfiltfilt, butter

import matplotlib.pyplot as plt

from featuresLib import *

# DEFINITIONS
USER = 'Mahsa'  # ['Mahsa', 'Jaimie']  # participant name
f_samp = 240 # smartwheel sampling frequency
IMU_samp = 200 # smartphone sampling frequency
cut_off = 5  # low-pass cut-off frequency
WIN_SIZE = 32  # window size

# IMU measurements and calculated wheelchair velocity
data_columns = ['AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel', 'Torque_L', 'Torque_R']
maneuvers = ['Obstacles15', 'Obstacles35', 'RampA', 'StraightF', 'Turn90FR', 'Turn90FL', 'Turn180L', 'Turn180R']
trials = ['T1', 'T2', 'T3']  # trial names

time_features = {'Mean': np.mean, 'Std': np.std, 'Max': np.amax, 'Min' : np.amin, 'RMS': rms}

# export data
EXPORT_PROCESSED_DATA = input('Export processed data? True/False? \n')
file_name_extension = '_WS' + str(WIN_SIZE) + '_' + USER + '.csv'
CURR_PATH = os.path.abspath('.')  # getting the current path of the notebook


def save_dic(path, dic):
    """
    function to save a dictionary of dataframes to csv files
    """
    for label, dataset in dic.items():
        filename = label + file_name_extension
        filename = os.path.join(path, filename)
        dataset.to_csv(filename, index=False)

def import_data(dataset_paths):
    # Import datasets as a dictionary of Pandas DataFrames
    datasets = {}

    for dataset_path in dataset_paths:
        # Parse labels from filenames
        dataset_label = os.path.split(dataset_path)[1].split('.')[0]

        # Read from XLS to Pandas
        dataset = pd.read_excel(dataset_path)

        # trim excessive datapoints for selected maneuvers
        if USER == 'Jaimie' and dataset_label == 'Turn180L_T1':
            dataset = dataset[:800]

        # update the dictionary of raw datasets
        datasets.update({dataset_label: dataset})

    # get a list of all imported maneuvers/trials
    dataset_labels = list(datasets.keys())

    return datasets, dataset_labels


def filt_func(datasetsR):
    # Filter each dataset individually
    datasetsF = {}

    for label, dataset in datasetsR.items():
        f_low = cut_off  # low-pass filter cut-off frequency
        w_low = f_low / (f_samp / 2)  # Get normalized frequencies [Nyquist frequecy = f_samp /2]
        sos = butter(N=2, Wn=w_low, btype='low', output='sos')  # Get  filter parameters (numerator and denominator)

        Time = dataset.pop('Time')
        cols = dataset.columns.tolist()
        dataset_copy = np.copy(dataset)

        for i in range(len(cols)):
            dataset_copy[:, i] = sosfiltfilt(sos, dataset_copy[:, i])

        df = pd.DataFrame(dataset_copy, columns=cols)
        df.insert(0, 'Time', Time)

        datasetsF.update({label: df})

    return datasetsF


def feature_func(datasetsF):
    """add new features to all filtered dataframes """

    featured_dic = {}

    def add_new_features(df):
        """ Appends L-R colummns to dataframe. If there is a column ending in "_L", there must be a "_R" too """
        for col in ['Torque_L', 'Torque_R']:
            df["Torque_sum"] = df['Torque_L'] + df['Torque_R']
            df["Torque_diff"] = df['Torque_R'] - df['Torque_L']

        return df

    def add_roc(df, dt):
        """ add rate of change of features (columns) in a dataframe"""
        df_roc = df[['Torque_L', 'Torque_R']].diff().fillna(0)
        df_roc.columns = ['Torque_L_roc', 'Torque_R_roc']
        return df.join(df_roc)

    for label, dataset in datasetsF.items():
        featured_dataset = add_new_features(dataset.copy())
        featured_dataset = add_roc(featured_dataset.copy(), 1 / f_samp)
        featured_dic.update({label: featured_dataset})

    return featured_dic


def segment_func(datasets_dic):
    """ function to create short windows"""

    datasets_seg = {}

    # Trim excess datapoints, then split into windows
    for label, dataset in datasets_dic.items():

        segmented_dataset = []
        dataset = dataset.drop(['Time'], axis=1)

        for i in range(int(len(dataset)/WIN_SIZE)):
            df_ = dataset.iloc[i*WIN_SIZE:(i+1)*WIN_SIZE, :]
            segmented_dataset.append(df_)

        datasets_seg.update({label: segmented_dataset})

    return datasets_seg


def feature_extraction(datasets, features_dic):

    """Extract given features from column of each dataset
       Converts a dictionary of datasets to a nested dictionary where each dataset has its own dictionary
       of axes/directions"""

    # will be updated with a nested dictionary of {label:{data column name:[dataframe with feature extracted columns]}}
    feat_datasets = {}

    # Calculate features for each window of each column of each dataset
    for label, dataset_list in datasets.items():

        # will be updated with keys as data columns (e.g., 'X Accel')
        cols_dic = {}

        # Loop over data columns
        for col in featured_columns:

            # will be updated with keys as extracted feature names (e.g., 'Mean')
            feats = {}

            def function_all_windows(function):
                featured_column = []

                for window in dataset_list:

                    # update a list of extracted feature for the ith column
                    featured_column.append(function(window[col]))

                return featured_column

            # Execute every function over all windows
            for feat_name, feat_func in features_dic.items():

                # apply feature extraction to the ith column for all windows
                feats.update({feat_name: function_all_windows(feat_func)})

            cols_dic.update({col: pd.DataFrame.from_dict(feats)})

        feat_datasets.update({label: cols_dic})

    return feat_datasets


def append_all_columns(columns, append_tag):

    """Append a tag to the end of every column name of a dataframe"""

    new_columns = []

    for column in columns:

        new_columns.append(column + ' ' + append_tag)

    return new_columns


def combine_extracted_columns(datasets):

    """Combined directions (axes) of a featured dataset"""

    combined_datasets = {}

    for label, dataset in datasets.items():
        # Get labels array of first column
        df_combined = pd.DataFrame()

        # Append direction name to feature name and combine everything in one frame
        for col_label, df in dataset.items():
            df_copy = pd.DataFrame(df)

            # Add direction and placement tags
            df_copy.columns = append_all_columns(df.columns, col_label)

            df_combined = df_combined.join(df, how='outer')

        combined_datasets.update({label: df_combined})

    return combined_datasets


dataset_paths = glob.glob(os.path.join(CURR_PATH, 'measurements', USER, '*.xls')) # data path
raw_datasets, dataset_labels = import_data(dataset_paths)  # import data
filt_datasets = filt_func(raw_datasets)  # Filtering Data
featured_datasets = feature_func(filt_datasets)  # Add new features (average, difference, rate of change of torque)
segmented_datasets = segment_func(featured_datasets)

featured_columns = segmented_datasets[dataset_labels[0]][0].columns.tolist()

time_featured_datasets = feature_extraction(segmented_datasets, time_features)  # Create array of features of each window for each dataset and direction
columned_time_feat_datasets = combine_extracted_columns(time_featured_datasets)  # Take time feature data and combine axes columns

# Saving feature extracted dataframes
if EXPORT_PROCESSED_DATA:
    path = os.path.join(CURR_PATH, 'Feature_Extracted_Data', USER, 'WinSize' + str(WIN_SIZE))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_dic(path, columned_time_feat_datasets)

print("SUCCESS!!!")

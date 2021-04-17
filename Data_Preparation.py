#!/usr/bin/env python
# coding: utf-8

# # IMPORTANT NOTES TO READ BEFORE RUNNING THE NOTEBOOK
# 
# ### The following instructions are related to the "notebook variables" section
# 
# 1. Determine the window size: **WIN_SIZE**
# 2. Determine the user name: **USER**
# 

# # Intention Detection Study: Data preprocessing and Visualization
# 
# Preprocessing and visualizing IMU and SmartWheel (SW) measurements. Comparing different trials of the same maneuver and comparing different maneuver dynamics & kinematics.
# 
# Creating the following dictionary of dataframes:
# - **raw datasets**: importing raw measurements from *Trimmed_Data*
# - **featured datasets**: adding extra features (average, diff, ratio, rate of change) using IMU/SW left/right measurements
# - **segmented datasets**: creating variable window sizes from featured datasets
# - **feature extracted datasets**: extracting features from time windows
# 
# Notebook outputs:
# - **Feature_Extracted_Data** is exported and will be used for clustering 
# - plotting raw vs. filtered measurements
# - plotting all torque/velocity measurements
# - plotting selected features/extracted-features

# ###  Importing relevant modules

# In[1]:


import os
import pathlib
import glob

import pandas as pd
import numpy as np

from scipy import signal
from scipy.signal import sosfiltfilt, butter

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Notebook variables

# In[2]:


# lowpass cut-off frequency
cut_off_list = [5, 8, 10, 15, 20] 
cut_off = cut_off_list[0]

# choose the window size
WIN_SIZE_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
WIN_SIZE = WIN_SIZE_list[7]

# Choose the study participant from the list
USER_list = ['Mahsa', 'Jaimie'] 
USER = USER_list[0]

# determine whether to export featured data or not
EXPORT_DATA = False


# ### Notebook parameters

# In[3]:


SW_samp = 240 # smartwheel's sampling frequency
IMU_samp = 200 # smartphone's sampling frequency

# original measurements and calculated chair velocity
data_columns =  ['AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel', 'Torque_L', 'Torque_R']

# Types of maneuvers, placements, and transforms used
maneuvers = ['Curb', 'Obstacles15', 'Obstacles35', 'RampA', 'RampD', 'StraightB', 'StraightF', 'Turn90BL',
            'Turn90BR', 'Turn90FR', 'Turn90FL', 'Turn180L', 'Turn180R']
# trial names
trials = ['T1', 'T2', 'T3']

# file name extension used when saving files
file_name_extension = '_WS'+ str(WIN_SIZE) + '_' + USER + '.csv' 


# ### Notebook functions

# In[4]:


# function to save a dictionary of dataframes to csv files
def save_dic(path, dic):
    '''
    function input: path to save csv files and dictionary of dataframes to be save
    '''
    for label, dataset in dic.items():
        filename = label + file_name_extension
        filename = os.path.join(path, filename)
        dataset.to_csv(filename, index=False)


# ## Part 1 - Import Torque and velocity data

# ### 1.1. Find xls files in the following 'Timmed_Data' directory

# In[5]:


# getting the current path of the notebook
CURR_PATH = os.path.abspath('.') 

# Glob all xls files in the folder
glob_paths = glob.glob(os.path.join(CURR_PATH, 'Trimmed_Data', USER, '*.xls'))

# Keep trials only and remove trim table files
dataset_paths = [path for path in glob_paths if 'Table' not in path]


# ### 1.2. Parsing data into Pandas

# In[6]:


# Import datasets as a dictionary of Pandas DataFrames
raw_datasets = {}

for dataset_path in dataset_paths:
    # Parse labels from filenames
    dataset_label = os.path.split(dataset_path)[1].split('.')[0]    

    # Read from XLS to Pandas
    dataset = pd.read_excel(dataset_path)
    
    # trim excessive datapoints for selected maneuvers
    if USER == 'Jaimie' and dataset_label == 'Turn180L_T1':
        dataset = dataset[:800]
    
#     # use a linear velocity threshold to cut excessive stationary measurements
#     if 'Turn180' not in dataset_label:
#         thresh_axes = 'Chair_LinVel' # indication of state of motion
#         STARTUP_THRESH = 0.1 # initial threshold for raw data
#         stop_index = dataset[dataset[thresh_axes] > STARTUP_THRESH].index[-1]
#         dataset = dataset[:stop_index]
    
    # update the dictionary of raw datasets
    raw_datasets.update({dataset_label: dataset})

# get a list of all imported maneuvers/trials
dataset_labels = list(raw_datasets.keys())


# In[7]:


# check the number of imported datasets and name of all maneuvers/trials
print('Number of raw datasets: {}'.format(len(dataset_labels)))
print('List of raw datasets: {}'.format(dataset_labels))


# In[8]:


# Check dataset formatting
raw_datasets[dataset_labels[0]].head()


# ### [OLD VERSION: Not using this anymore] Resample and create resampled_datasets 
# #### Function to downsample SW data and trim signals to have the same time duration

# In[9]:


def resample_(datasets):
    
    ''' 
    input: dictionary of datasets to resample 
    
    determining the length (time duration) of IMU/SW measurements, trim accordingly & adding 'Time' column.
        
    return a dictionary of dataframes. Each dataframe has features with similar length and sampling frequency (200Hz)
    
    '''
        
    resampled_datasets={}
    
    for label, dataset in datasets.items():
        dataset_copy = dataset.copy()
        resampled_dataset = pd.DataFrame(columns=dataset.columns.tolist())
        
        valid_indx_IMU = dataset_copy['Chair_LinVel'].last_valid_index()
        IMU_time = np.linspace(0,valid_indx_IMU/IMU_samp, num = valid_indx_IMU)
        
        valid_indx_SW_L = dataset_copy['Torque_L'].last_valid_index()
        valid_indx_SW_R = dataset_copy['Torque_R'].last_valid_index()
        valid_indx_SW = min(valid_indx_SW_L, valid_indx_SW_R)
        SW_time = np.linspace(0,valid_indx_SW/SW_samp, num = valid_indx_SW)
        
        # finding the shortest measurement (IMU or SW)
        if SW_time[-1] > IMU_time[-1]:
            for col in IMU_data:
                resampled_dataset[col] = dataset_copy[col][:valid_indx_IMU].copy()
                Time = IMU_time.copy()

            for col in SW_data:
                resamp_num = int(SW_time[-1] * IMU_samp)
                col_copy = signal.resample(dataset_copy[col], resamp_num)
                resampled_dataset[col] = col_copy[:valid_indx_IMU]
            
        elif SW_time[-1] < IMU_time[-1]:
            for col in SW_data:
                resamp_num = int(SW_time[-1] * IMU_samp)
                col_copy = dataset_copy[col][:valid_indx_SW]
                resampled_dataset[col] = signal.resample(col_copy, resamp_num)
                
            for col in IMU_data:
                valid_len = min(valid_indx_IMU, resamp_num)
                resampled_dataset[col] = dataset_copy[col][:valid_len]
                Time = IMU_time[:valid_len]

        resampled_dataset.insert(0, 'Time', Time)
        resampled_datasets.update({label: resampled_dataset})
        
    return resampled_datasets


# ## Part 2 - Filtering Data

# ### 2.1. Butterworth filter implementation

# In[10]:


# Filter each dataset individually
filt_datasets = {}

for label, dataset in raw_datasets.items():
    # Sampling rates are not consistent across all datasets
    f_samp = max(SW_samp, IMU_samp)  # sampling frequency (all data upsampled)
    f_low = cut_off # lowpass filter cut-off frequency
    
    # Get normalized frequencies
    w_low = f_low / (f_samp / 2) # Nyquist frequecy = f_samp /2 

    # Get Butterworth filter parameters (numerator and denominator)
    ## The function sosfiltfilt (and filter design using output='sos') should be preferred over filtfilt for most 
    ## filtering tasks, as second-order sections have fewer numerical problems.
    sos = butter(N=2, Wn=w_low, btype='low', output='sos')
    
    # Number of columns containing data
    n_data_col = len(data_columns) # not counting the 'Time' column
    
    # Filter all the data columns
    Time = dataset.pop('Time')
    cols = dataset.columns.tolist()
    dataset_copy = np.copy(dataset)
    
    for i in range(n_data_col):
        dataset_copy[:, i] = sosfiltfilt(sos, dataset_copy[:, i])
    
    df = pd.DataFrame(dataset_copy, columns=cols)
    df.insert(0, 'Time', Time)
        
    filt_datasets.update({label: df})


# In[11]:


# Check filtered dataframes
print('Number of datasets in filtered dictionary {}'.format(len(filt_datasets)))
filt_datasets[dataset_labels[0]].head()


# ## Part 3 -  Plotting experimental measurements

# ### 3.1. compare filtered vs. raw datasets

# In[12]:


# compare filt and resampled-raw data
def compare_filt_raw(maneuver):
    nrow = len(data_columns)
    ncol = 1
    
    fontsize = 14
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(10,20), constrained_layout=True)
    axs = axs.ravel()
    
    fig.suptitle(maneuver, fontsize= fontsize *1.2)
    
    for i, col in enumerate(data_columns):   
        
        axs[i-1].plot(raw_datasets[maneuver][col], label='raw')
        axs[i-1].plot(filt_datasets[maneuver][col], label = 'filt')
        axs[i-1].legend(fontsize = fontsize)
        axs[i-1].set_ylabel(col, fontsize = fontsize)


# In[13]:


# # compare raw and filtered data 
# maneuver = dataset_labels[1]
# compare_filt_raw(maneuver)


# ### 3.2. plot all filtered measurements

# In[14]:


# plotting left and right wheels' torque and angular velocity + chair's linear and angular velocity

def plot_allData_overlay(datasets, Torque_AngVel_only = False):
    
    ''' 
    datasets: dictionary of datasets of the same maneuver to plot 
    
    plotting left/right torque, left/right angular velocity, linear/angular velocity of the chair 
    for different trials of a certain maneuver
    
    '''
    if Torque_AngVel_only:
    
        axes = ['Torque_L', 'Torque_R', 'AngVel_L', 'AngVel_R']

        ncol = 2
        nrow = 1

        for label, dataset in datasets.items():
            fig, axs = plt.subplots(nrow, ncol, figsize=(15,6), constrained_layout=True, sharex=True)
            axs = axs.ravel()

            for i in range(2):           
                if 'Torque' in axes[2*i]:
                    axs[i].plot(dataset['Time'], dataset[axes[2*i]], label='Left wheel')
                    axs[i].plot(dataset['Time'], dataset[axes[2*i+1]], label='Right wheel')
                    axs[i].set_ylabel('Torque (Nm)',fontsize = 20)
                    axs[i].set_xlabel('Time (s)',fontsize = 20)
                    axs[i].tick_params(direction='out', labelsize = 20)
                    axs[i].legend(fontsize = 18)
                    axs[i].grid()

                elif 'Ang' in axes[2*i]:
                    axs[i].plot(dataset['Time'], dataset[axes[2*i]], label='Left wheel')
                    axs[i].plot(dataset['Time'], dataset[axes[2*i+1]], label='Right wheel')
                    axs[i].set_ylabel('Angular velocity (rad/s)',fontsize = 20)
                    axs[i].set_xlabel('Time (s)',fontsize = 20)
                    axs[i].tick_params(direction='out', labelsize = 20)
                    axs[i].legend(fontsize = 18, loc='lower right')
                    axs[i].grid()

                    # set the x lim
                    axs[i].set_xlim(0, dataset["Time"].iloc[-1])

                fig.suptitle(label, fontsize = 30, verticalalignment = 'top')  
                
#             save_name = os.path.join(CURR_PATH, 'imgs',  label + '_' + USER + '_Torque_AngVel_only' + '.png')
#             plt.savefig(save_name)
                
    else:
        axes = ['Torque_L', 'Torque_R', 'AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel']

        ncol = 1
        nrow = 3

        for label, dataset in datasets.items():
            fig, axs = plt.subplots(nrow, ncol, figsize=(10,10), constrained_layout=True, sharex=True)
            axs = axs.ravel()

            for i in range(3):           
                if 'Torque' in axes[2*i]:
                    axs[i].plot(dataset['Time'], dataset[axes[2*i]], label=axes[2*i])
                    axs[i].plot(dataset['Time'], dataset[axes[2*i+1]], label=axes[2*i+1])
                    axs[i].set_ylabel('Nm')
                    axs[i].legend()
                    axs[i].grid()

                elif 'Ang' in axes[2*i]:
                    axs[i].plot(dataset['Time'], dataset[axes[2*i]], label=axes[2*i])
                    axs[i].plot(dataset['Time'], dataset[axes[2*i+1]], label=axes[2*i+1])
                    axs[i].set_ylabel('rad/s')
                    axs[i].legend()
                    axs[i].grid()

                elif 'Lin' in axes[2*i]:
                    axs2 = axs[i].twinx()  # instantiate a second axes that shares the same x-axis
                    lns1 = axs[i].plot(dataset['Time'], dataset[axes[2*i]], label=axes[2*i])
                    lns2 = axs2.plot(dataset['Time'], dataset[axes[2*i+1]], label=axes[2*i+1], color ='#ff7f0e')
                    axs[i].set_ylabel('m/s')
                    axs[i].set_xlabel('Time(s)')
                    axs2.set_ylabel('rad/s')

                    lns = lns1+lns2
                    labs = [l.get_label() for l in lns]
                    axs[i].legend(lns, labs, loc=0)                
                    axs[i].grid()

                    # set the x lim
                    axs[i].set_xlim(0, dataset["Time"].iloc[-1])

                fig.suptitle(label, fontsize = 14, verticalalignment = 'top')
            
#             save_name = os.path.join(CURR_PATH, 'imgs',  label + '_' + USER + '.png')
#             plt.savefig(save_name)

    plt.show()


# In[15]:


# # plot kinematic and kinetic measurements of all trials
# # determine whether to plot torqu/angular veloicty only or showing linear velocity of the wheelchair too
# for maneuver in maneuvers:
#     datasets_to_plot = {label: dataset for label, dataset in filt_datasets.items() 
#                         if maneuver in label}
#     plot_allData_overlay(datasets_to_plot, Torque_AngVel_only = True)


# ## Part 4 -  Add new features (average, difference, ratio, rate of change of left/right torque)

# ### 4.1. Functions to claculate new features & rate of change

# In[16]:


''' Appends L-R colummns to dataframe. If there is a column ending in "_L", there must be a "_R" too '''
def add_new_features (df):

    # For dealing with division by 0 when taking ratio of L/R data
    RATIO_OFFSET = 1 
    
    for col in ['Torque_L','Torque_R']:
        df["Torque_sum"] = df['Torque_L'] + df['Torque_R']
        df["Torque_diff"] = df['Torque_R'] - df['Torque_L']         
#         df.loc[abs(df.Torque_L) > abs(df.Torque_R), 'Torque_ratio'] = df['Torque_R']/(df['Torque_L'] + RATIO_OFFSET)
#         df.loc[abs(df.Torque_R) > abs(df.Torque_L), 'Torque_ratio'] = df['Torque_L']/(df['Torque_R'] + RATIO_OFFSET)
    
    return df


# In[17]:


'''add rate of change of features (columns) in a dataframe'''
def add_roc(df, dt):
    df_roc = df[['Torque_L','Torque_R']].diff().fillna(0)
    df_roc.columns = ['Torque_L_roc', 'Torque_R_roc']
    return df.join(df_roc)      


# ### 4.2. Add new features to filtered dataframes

# In[18]:


# add new features to all filtered dataframes
featured_datasets = {}
for label, dataset in filt_datasets.items():
    featured_dataset = add_new_features(dataset.copy())
    featured_dataset = add_roc(featured_dataset.copy(), 1/f_samp)
    featured_datasets.update({label: featured_dataset})
    
# Get a list of all features
feat_columns = featured_datasets[dataset_labels[0]].columns.tolist()


# In[19]:


# Check featured dataframes
featured_datasets['Obstacles15_T1'].head()


# ### 4.3. Visualize torque features for all trials

# In[20]:


''' visualise added features '''
def plt_new_feats(label, df, features):
   
    fontsize = 14
    
    fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
    
    fig.suptitle(label, fontsize= fontsize *1.2)
    
    for feat in features:
        if 'avg' in feat:
            linewidth = 3
        elif 'diff' in feat:
            linewidth = 3
        elif 'roc' in feat:
            linewidth = 3
        else:
            linewidth = 5
            
        ax.plot(df[feat], label = feat, linewidth = linewidth)
        ax.legend(ncol = 3, fontsize = fontsize)
        ax.set_ylabel('Torque (Nm)', fontsize = fontsize)


# In[21]:


# # uncomment to examine different feature sets
# for label , dataset in featured_datasets.items():
#     plt_new_feats(label, dataset, ['Torque_L', 'Torque_R',  'Torque_sum', 'Torque_diff', 'Torque_L_roc', 'Torque_R_roc'])


# ## Part 5 - Data segmentation

# In[22]:


segmented_datasets = {}

# Trim excess datapoints, then split into windows
for label, dataset in featured_datasets.items():
    
    segmented_dataset = []
    
    dataset = dataset.drop(['Time'], axis=1)
    
    for i in range(int(len(dataset)/WIN_SIZE)):
        df_ = dataset.iloc[i*WIN_SIZE:(i+1)*WIN_SIZE,:] 
        segmented_dataset.append(df_)
    
    segmented_datasets.update({label: segmented_dataset})


# In[23]:


# Check if its constructed correctly
print('Num windowed datasets: {}'.format(len(segmented_datasets)))
print('Num of windows in first dataset: {}'.format(len(segmented_datasets[dataset_labels[0]])))
print('Shape of individual window: {}'.format(segmented_datasets[dataset_labels[0]][-1].shape))


# In[24]:


# to verify original and generated datasets window sizes
for name in segmented_datasets.keys():
    print('Dataset  {}: No. of data points {} & No. of windows {} '.format(name, len(featured_datasets[name]), len(segmented_datasets[name])))


# In[25]:


# Check a window of the first dataframe
segmented_datasets[dataset_labels[0]][0].tail()


# In[26]:


featured_columns = segmented_datasets[dataset_labels[0]][0].columns.tolist()
featured_columns


# ## Part 6 - Feature Engineering 

# ### 6.1. Feature extraction functions

# In[27]:


# Feature extraction functions

'''L2 norm of an array'''
def l2norm(array):
    return np.linalg.norm(array, ord=2)

'''Root mean squared of an array'''
def rms(array):
    return np.sqrt(np.mean(array ** 2))


# In[29]:


def feature_extraction(datasets, features_dic):
    
    '''Extract given features from column of each dataset
       Converts a dictionary of datasets to a nested dictionary where each dataset has its own dictionary
       of axes/directions'''
    
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


# ### 6.2. Defining time_features 

# In[30]:


# Time domain feature functions and names
## older version of time features
# time_features = {'Mean': np.mean, 'Std': np.std,  'Norm': l2norm, 
#                  'Max': np.amax, 'Min' : np.amin, 'RMS': rms, 'Sum': np.sum} 

# removed redundent time features
time_features = {'Mean': np.mean, 'Std': np.std, 
                 'Max': np.amax, 'Min' : np.amin, 'RMS': rms} 


# ### 6.3. Creating a dictionary of feature extracted datasets

# In[31]:


# Create array of features of each window for each dataset and direction
time_featured_datasets = feature_extraction(segmented_datasets, time_features)


# In[32]:


# Check if feature data is constructed correctly and print some info
print('Num datasets: {}'.format(len(time_featured_datasets)))
print('Num directions: {}'.format(len(time_featured_datasets[dataset_labels[0]])))
print('Shape of first dataset first direction: {}'.format(time_featured_datasets[dataset_labels[0]]['Torque_L'].shape))


# In[33]:


time_featured_datasets[dataset_labels[0]]['Torque_L'].head()


# In[34]:


# # examine scaled/original extracted features for all datasets
# for label_ in dataset_labels:
#     for feat in featured_columns:  
#         df_test = time_featured_datasets[label_][feat]
#         scaled_features = StandardScaler().fit_transform(df_test.values)
#         scaled_features_df = pd.DataFrame(scaled_features, index=df_test.index, columns=df_test.columns)
        
#         for col in list(time_features.keys())[:-1]:
#             print(label_, feat, col)
#             if col is not 'Norm':
#                 plt.plot(df_test[col], label = col)
#         plt.legend()    
#         plt.show()


# ## Part 7 - Columning Data 

# In[35]:


def append_all_columns(columns, append_tag):

    '''Append a tag to the end of every column name of a dataframe'''

    new_columns = []
    
    for column in columns:
        
        new_columns.append(column + ' ' + append_tag)
        
    return new_columns


# In[36]:


def combine_extracted_columns(datasets):

    '''Combined directions (axes) of a featured dataset'''

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


# In[37]:


# Take time feature data and combine axes columns
columned_time_feat_datasets = combine_extracted_columns(time_featured_datasets)


# In[38]:


# get a list of extracted features names
feat_data_columns = columned_time_feat_datasets[dataset_labels[0]].columns.tolist()
print(feat_data_columns)


# In[39]:


# Confirm formatting
columned_time_feat_datasets[dataset_labels[0]].head()


# In[40]:


# # compare original and scaled extracted features
# #create an empty dataframe with columns similar to the imported datasets
# df_test = pd.DataFrame(columns=columned_time_feat_datasets[dataset_labels[0]].columns.tolist())
# scaler = StandardScaler()

# # combine desired datasets into one dataframe
# for label in dataset_labels:
#     df_test = pd.concat([df_test, columned_time_feat_datasets[label]], ignore_index=True)

# df_test_stand = scaler.fit_transform(df_test.copy())
# df_test_stand = pd.DataFrame(df_test_stand, columns=feat_data_columns)

# for feat in feat_data_columns:
#     plt.plot(df_test[feat], label = 'measurements')
#     plt.plot(df_test_stand[feat], label = 'normalized')
#     plt.ylabel(feat); plt.legend();
#     plt.show()


# ## Part 8 - Saving feature extracted dataframes

# In[41]:


path = os.path.join(CURR_PATH, 'Feature_Extracted_Data', USER, 'WinSize'+ str(WIN_SIZE))
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
save_dic(path, columned_time_feat_datasets)


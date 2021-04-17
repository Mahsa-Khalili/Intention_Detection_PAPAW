#!/usr/bin/env python
# coding: utf-8

# # Intention Detection Classification Pipeline
# 
# - **Import notebook dependencies**
# - **Defining notebook variables**
# - **Defining notebook parameters **
# - **Feature correlation analysis**
# - **Creating the classification pipeline**
#     - data normalization
#     - features selection
#     - classification
#     - gridsearch to optimize hyperparameters
#     - export model
# - **Model evaluation**
#     - examine model accuray, confusion matrix
# - **Summary of the gridsearch results**
#     - best model parameters
#     - best selected features

# ### Import dependencies

# In[ ]:


import os
import pandas as pd
import numpy as np
from scipy import stats
import glob
import csv
import time

import random
from random import randrange

# plotting
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import plot_confusion_matrix

# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# feature selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
# from yellowbrick.model_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# cross-validation
from sklearn.model_selection import KFold

# pipeline 
from sklearn.pipeline import Pipeline

# grid search
from sklearn.model_selection import GridSearchCV
## explicitly require this experimental feature
# from sklearn.experimental import enable_halving_search_cv 
# from sklearn.model_selection import HalvingGridSearchCV

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from mlxtend.evaluate import PredefinedHoldoutSplit # check whether this is used

# import/export
import joblib
from joblib import dump, load


# ### Notebook Variables

# In[ ]:


# determine window size
WIN_SIZE_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
WIN_SIZE = WIN_SIZE_list[4]

# determine the study participant
USER_list = ['Mahsa', 'Jaimie']
USER = USER_list[0]

# choose the feature subsets for clustering
featureSet_list = ['ALL', 'ALL_TORQUE', '2D_TORQUE', 'LR_TORQUE', 'LR_TORQUE_MEAN', '2D_TORQUE_MEAN'] 
featureSet = featureSet_list[1]

# number of clusters
n_components_list = ['4', '5', '6', '7', '8']
n_components = n_components_list[2]


# ### Notebook Parameters (Constant values)

# In[ ]:


# path to save labeled data and corresponding figures
CURR_PATH = os.path.abspath('.')

# create a color pallette
cmap = matplotlib.cm.get_cmap('tab10')

# list of all maneuvers
maneuvers = ['Obstacles15', 'Obstacles35', 'RampA', 'StraightF', 'Turn90FR', 'Turn90FL', 'Turn180L', 'Turn180R']

# trials
trials = ['T1', 'T2', 'T3']

# unused columns for correlation analysis
unused_cols = ['maneuver', 'trial', 'Prob_L0', 'Prob_L1', 'Prob_L2', 'Prob_L3', 'Prob_L4', 'Prob_L5']

# LABELS
Labels = ['0', '1', '2', '3', '4', '5']


# ## Part 1 - Importing Datasets

# ### 1.2. Import data 

# In[ ]:


# get path to train datasets                         
glob_paths = glob.glob(os.path.join(CURR_PATH, 'Labeled_Data', USER, str(WIN_SIZE), '*.csv'))

# Read data and update current user dictionary
labeled_df = pd.read_csv(glob_paths[0])

labeled_df['Clus_label'] = labeled_df['Clus_label'].astype(float)


# In[ ]:


# Check some data
labeled_df.head()


# In[ ]:


velocity_columns = [col for col in labeled_df.columns.tolist() if 'Vel' in col]
torque_columns = [col for col in labeled_df.columns.tolist() if 'Torque' in col]


# ### 1.3. Split train/test data

# In[ ]:


random.seed(1)

test_trials = []

test_df = pd.DataFrame(columns = labeled_df.columns)
train_df = pd.DataFrame(columns = labeled_df.columns)

for maneuver in maneuvers:
    
    if 'Ramp' in maneuver and USER == 'Mahsa':
        test_tr = trials[randrange(2)]
    else:
        test_tr = trials[randrange(3)]
    
    test_trials.append(maneuver+'_'+test_tr)
        
    df_TE = pd.DataFrame(columns = labeled_df.columns)
    df_TE = labeled_df.loc[labeled_df['trial']== maneuver + '_' + test_tr + '_WS' + str(WIN_SIZE) + '_' + USER]
    test_df = test_df.append(df_TE, ignore_index=True)
    
    train_tr = [tr for tr in trials if tr != test_tr]
    
    for tr in train_tr:
        df_TR = pd.DataFrame(columns = labeled_df.columns)
        df_TR = labeled_df.loc[labeled_df['trial']== maneuver + '_' + tr + '_WS' + str(WIN_SIZE) + '_' + USER]
        train_df = train_df.append(df_TR, ignore_index=True)


# In[ ]:


test_trials


# In[ ]:


print('test set shape: {}, train set shape: {}'.format(test_df.shape, train_df.shape))


# ## Part 2 - Feature correlation analysis 

# In[ ]:


# create directory to save results
results_path = os.path.join(CURR_PATH, 'Results')

# get current time to use for saving models/figures
timestr = time.strftime("%Y%m%d-%H%M%S")

# create directiry for the current time
path_ = os.path.join(results_path, timestr)
os.makedirs(path_) 


# In[ ]:


def correlation_analysis(dataset, target_label):
      
    df = dataset.copy()
    
    df = df.drop(columns=unused_cols)
        
    # calculate correlation matrix
    cor = df.corr()
    
    display(cor.head())
    
    # calculate correlation with output variable
    cor_target = abs(cor[target_label])
    cor_target = cor_target.sort_values(ascending=False)
    print('feature correlation values with {} target value: \n{}'.format(target_label, cor_target))
    
    # save correlation values to csv
    filename = 'correlation_matrix.csv'
    filename = os.path.join(path_, filename)
    cor_target.to_csv(filename)
    
    # drop columns associated with data labels
    cor = cor.drop(['Clus_label'], axis = 1)
    cor = cor.drop(['Clus_label'], axis = 0)
    
    # plot heat map
    plt.figure(figsize = (16,12))
    sns.heatmap(abs(cor), cmap = plt.cm.Reds)
    
    # save correlation heatmap
    fig_name = 'correlation_analysis.jpg'
    fig_name = os.path.join(path_, fig_name)
    plt.savefig(fig_name)


# In[ ]:


correlation_analysis(train_df, 'Clus_label')


# ## Part 3 - Classification Pipeline

# ### 3.1. Create train/test datasets

# In[ ]:


## Function to Create TRAIN/VALIDATE/TEST data sets
def create_train_test(dataset, target_label, test_size):  
    '''
    input: get dataset and desired target label to perform classification
    output: train/test splits
    '''
    df = dataset.copy()
    
    df = df.drop(columns=unused_cols)
    
    y = df.pop(target_label)
        
    # split data
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=0, shuffle=False)
        
    return X_train, X_test, y_train, y_test


# In[ ]:


# create train set
X_train, X_, y_train, y_ = create_train_test(train_df, 'Clus_label', test_size = 1)

print('X_train shape = {} , y_train shape = {}'.format(X_train.shape, y_train.shape))
print('X_ shape = {} , y_ shape = {}'.format(X_.shape, y_.shape))


# In[ ]:


# create test set
X_, X_test, y_, y_test = create_train_test(test_df, 'Clus_label', test_size = len(test_df)-1)

print('X_ shape = {} , y_ shape = {}'.format(X_.shape, y_.shape))
print('X_test shape = {} , y_test shape = {}'.format(X_test.shape, y_test.shape))


# In[ ]:


# X_train = X_train[torque_columns]
# X_test = X_test[torque_columns]

# print('X_train shape = {} , y_train shape = {}'.format(X_train.shape, y_train.shape))
# print('X_test shape = {} , y_test shape = {}'.format(X_test.shape, y_test.shape))


# ### 3.2. Create Classification pipeline

# In[ ]:


def clf_pipeline(X_train, y_train, HalvGrid = False):
    
    # cross validation nfolds
    cv = 5
    
    # classifier to use for feature selection
    feat_selection_clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state = 0)
    
    # create pipeline
    pipe = Pipeline([('scaler', StandardScaler()), 
                     ('selector', SFS(estimator = feat_selection_clf, 
                                      k_features=(2, X_train.shape[1]), 
                                      forward=True, 
                                      floating=False, 
                                      scoring='accuracy', 
                                      cv=cv, 
                                      n_jobs=-1)),
                     ('classifier', RandomForestClassifier())])
    
    # set parameter grid
    param_grid = [
        {'selector':[SFS(estimator = feat_selection_clf)],
                        'selector__estimator__n_estimators':[50],
                        'selector__estimator__max_depth':[3,5,10]},

#         {'selector':[RFE(estimator= feat_selection_clf)],
#                         'selector__estimator__n_estimators':[5, 10],
#                         'selector__estimator__max_depth':[3,4], 
#          'selector__n_features_to_select':[1,2]},
        
#         {'selector':[RFECV(estimator = feat_selection_clf, min_features_to_select = 1)],
#                          'selector__estimator__n_estimators':[5, 10],
#                          'selector__estimator__max_depth':[3,4]},
        
        {'classifier':[RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state = 0)],
         'classifier__n_estimators':[30,50],
         'classifier__max_depth':[5,10,15]}]
    
    # dictionary of evaluation scores
    scoring = {
        'precision_score': make_scorer(precision_score, average='macro'),
        'recall_score': make_scorer(recall_score, average='macro'),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score':make_scorer(f1_score, average='macro')}

    # gridsearch 
    if HalvGrid:
        # HalvingGridSearch
        grid = HalvingGridSearchCV(pipe, 
                                   param_grid, 
                                   cv = cv, 
                                   scoring = scoring,
                                   refit ='accuracy_score',
                                   return_train_score=True,
                                   random_state = 0,
                                   n_jobs = -1,
                                   verbose = 0)
    else:
        grid = GridSearchCV(pipe, 
                            param_grid, 
                            scoring = scoring,
                            n_jobs = -1,
                            refit ='accuracy_score',
                            cv = cv, 
                            verbose = 0,
                            return_train_score=True)
    
    grid.fit(X_train, y_train)
    
    return grid 


# In[ ]:


# run the classification pipeline
model_ = clf_pipeline(X_train, y_train)


# In[ ]:


# get parameters of the best model
print("Best estimator via GridSearch \n", model_.best_estimator_)


# ### 3.3. Export Model

# In[ ]:


# model name/directory
model_name = 'model.joblib'
model_name = os.path.join(path_, model_name)

# dump model
joblib.dump(model_, model_name)


# ## Part 4 - Evaluation

# ### 4.1. Evaluation score

# In[ ]:


# accuracy
print('Accuracy score = {:0.2f}'.format(model_.score(X_test, y_test) * 100, '.2f'))

y_pred = model_.predict(X_test)

# f1_score
f_score = f1_score(y_test, y_pred, average = 'macro')* 100
print('F1-score = {:0.2f}'.format(f_score))

# recall
recall_ = recall_score(y_test, y_pred, average = 'macro')* 100
print('Recall score = {:0.2f}'.format(recall_))


# ### 4.2. Confusion matrix

# In[ ]:


# confusion matrix
title = "Normalized confusion matrix"

disp = plot_confusion_matrix(model_, X_test, y_test, display_labels=Labels,
                             cmap=plt.cm.Blues,
                             normalize='true',
                             xticks_rotation = 45,
                             values_format = '.2f')
disp.ax_.set_title(title)
disp.ax_.grid(False)
disp.figure_.set_size_inches(12,12)

# save confusion matrix
fig_name = 'confusion_matrix.jpg'
fig_name = os.path.join(path_, fig_name)
plt.savefig(fig_name)


# ### 4.3. Analyze computational performance

# In[ ]:


# # computational performance
# X_Test_test = X_test[:1].copy()

# # method 1
# %timeit y_pred = model_.predict(X_Test_test)

# # method 2
# time1 = time.time()
# y_pred = model_.predict(X_Test_test)
# time2 = time.time()
# print('prediction time: {} ms'.format((time2-time1)*1000))


# ## Part 5 - GridSearch Summary

# ### 5.1. Model best score

# In[ ]:


# Mean cross-validated score of the best_estimator
print('Best feature combination had a CV accuracy of:', model_.best_score_)


# ### 5.2. Model best parameters

# In[ ]:


print("Best parameters via GridSearch \n", model_.best_params_)


# ### 5.3. Visualize GridSearch results
# ##### Source: https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search

# In[ ]:


# def GridSearch_table_plot(grid_clf, param_name,
#                           num_results=4,
#                           negative=True,
#                           graph=True,
#                           display_all_params=False):

#     '''Display grid search results

#     Arguments
#     ---------

#     grid_clf           the estimator resulting from a grid search
#                        for example: grid_clf = GridSearchCV( ...

#     param_name         a string with the name of the parameter being tested

#     num_results        an integer indicating the number of results to display
#                        Default: 15

#     negative           boolean: should the sign of the score be reversed?
#                        scoring = 'neg_log_loss', for instance
#                        Default: True

#     graph              boolean: should a graph be produced?
#                        non-numeric parameters (True/False, None) don't graph well
#                        Default: True

#     display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
#                        Default: True

#     Usage
#     -----

#     GridSearch_table_plot(grid_clf, "min_samples_leaf")

#                           '''
#     from matplotlib      import pyplot as plt
#     from IPython.display import display
#     import pandas as pd

#     clf = grid_clf.best_estimator_
#     clf_params = grid_clf.best_params_
#     if negative:
#         clf_score = -grid_clf.best_score_
#     else:
#         clf_score = grid_clf.best_score_
    
#     clf_stdev = grid_clf.cv_results_['std_test_accuracy_score'][grid_clf.best_index_]
#     cv_results = grid_clf.cv_results_

#     print("best parameters: {}".format(clf_params))
#     print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
#     if display_all_params:
#         import pprint
#         pprint.pprint(clf.get_params())

#     # pick out the best results
#     # =========================
#     scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_accuracy_score')

#     best_row = scores_df.iloc[0, :]
#     if negative:
#         best_mean = -best_row['mean_test_accuracy_score']
#     else:
#         best_mean = best_row['mean_test_accuracy_score']
#     best_stdev = best_row['std_test_accuracy_score']
#     best_param = best_row['param_' + param_name]

#     # display the top 'num_results' results
#     # =====================================
#     display(pd.DataFrame(cv_results) \
#             .sort_values(by='rank_test_accuracy_score').head(num_results))

#     # plot the results
#     # ================
#     scores_df = scores_df.sort_values(by='param_' + param_name)

#     if negative:
#         means = -scores_df['mean_test_accuracy_score']
#     else:
#         means = scores_df['mean_test_accuracy_score']
#     stds = scores_df['std_test_accuracy_score']
#     params = scores_df['param_' + param_name]
    
#     # plot
#     if graph:
#         plt.figure(figsize=(8, 8))
#         plt.errorbar(params, means, yerr=stds)

#         plt.axhline(y=best_mean + best_stdev, color='red')
#         plt.axhline(y=best_mean - best_stdev, color='red')
#         plt.plot(best_param, best_mean, 'or')

#         plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
#         plt.xlabel(param_name)
#         plt.ylabel('Score')
#         plt.show()


# In[ ]:


# GridSearch_table_plot(model_, "selector__estimator__n_estimators", negative=False)


# ### 5.4. Gridsearch evaluation - multiple scores
# ##### source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

# In[ ]:


# def score_evaluation(grid, param):

#     # plotting the results
#     results = grid.cv_results_
    
#     scoring = {
#             'precision_score': make_scorer(precision_score, average='macro'),
#             'recall_score': make_scorer(recall_score, average='macro'),
#             'accuracy_score': make_scorer(accuracy_score),
#             'f1_score':make_scorer(f1_score, average='macro')}

#     plt.figure(figsize=(10, 8))
#     plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
#               fontsize=16)

#     plt.xlabel(param)
#     plt.ylabel("Score")

#     ax = plt.gca()
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0.0, 1)

#     # Get the regular numpy array from the MaskedArray
#     X_axis = np.array(results['param_selector__estimator__n_estimators'].data, dtype=float)

#     for scorer, color in zip(sorted(scoring), ['g', 'k','r','b']):
#         for sample, style in (('train', '--'), ('test', '-')):
#             sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
#             sample_score_std = results['std_%s_%s' % (sample, scorer)]
#             ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                             sample_score_mean + sample_score_std,
#                             alpha=0.1 if sample == 'test' else 0, color=color)
#             ax.plot(X_axis, sample_score_mean, style, color=color,
#                     alpha=1 if sample == 'test' else 0.7,
#                     label="%s (%s)" % (scorer, sample))

#         best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
#         best_score = results['mean_test_%s' % scorer][best_index]

#         # Plot a dotted vertical line at the best score for that scorer marked by x
#         ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#                 linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

#         # Annotate the best score for that scorer
#         ax.annotate("%0.2f" % best_score,
#                     (X_axis[best_index], best_score + 0.005))

#     plt.legend(loc="best")
#     plt.grid(False)
#     plt.show()


# In[ ]:


# score_evaluation(model_, "classifier__max_depth")


# ### 5.5. Gridsearch report

# In[ ]:


# results = model_.cv_results_
# results


# ## Part 6 - Feature selection summary

# ### 6.1. Feature selection score & best features

# In[ ]:


# get prediction score of best selected features
print('\nFeature selection score: {}'.format(model_.best_estimator_['selector'].k_score_))


# In[ ]:


# get best features
best_feats_idx = model_.best_estimator_['selector'].k_feature_idx_;

best_feats = list(X_test.columns[list(best_feats_idx)].values.tolist()) 

print('\nBest features: \n{}'.format(best_feats))

# save a list of selected features
filename = 'selected_features.csv'
filename = os.path.join(path_, filename)

with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows([c.strip() for c in r.strip(', ').split(',')] for r in best_feats)


# In[ ]:


# test shape of the feature transformed dataframe
X_test_transformed = model_.best_estimator_['selector'].transform(X_test)
print('shape of the transformed dataset with best features: {}'.format(X_test_transformed.shape))


# ### 6.2. Visualize feature selection scores

# In[ ]:


# plotting feature selection characteristics
plot_sfs(model_.best_estimator_['selector'].get_metric_dict(), kind='std_err', figsize=(12,5))
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid(b=True, which='major', axis='both')

# save confusion matrix
fig_name = 'feature_selection.jpg'
fig_name = os.path.join(path_, fig_name)
plt.savefig(fig_name)


# In[ ]:


# finding the index range for each maneuver
def range_dic_(df_):
    """
    get the start index of each maneuver from the original dataframe
    """
    range_dic = {}
    for man in df_['maneuver']:
        trial_indx = df_.index[df_['maneuver'] == man].tolist()
        range_ = (min(trial_indx), max(trial_indx))
        range_dic.update({man:range_})
    return range_dic


# In[ ]:


man_range = range_dic_(test_df)
man_range


# In[ ]:


# function to plot clusters in time series data
def plt_ts_cluster_prediction(df_clus_, predictions, clusterNum, features_to_plot, man_type = 'All', zoom = False):
    """
    input: input original dataframe (with maneuver columns), clustered dataframe, number of clusteres, 
           and selected features to plot
    output: plotting clustered time series data with different colors
    """
    
    MARKER_SIZE =10
    LINE_WIDTH = 5
    color_dict = {}
    
    plt_num = len(features_to_plot)
    
    fig, axs = plt.subplots(plt_num, 1, figsize=(15,15), constrained_layout=True)
    axs = axs.ravel()
    
    if man_type != 'All':
        df_clus = df_clus_[man_range[man_type][0]:man_range[man_type][1]]

    true_states = df_clus['Clus_label'].astype(int)
    predicted_states = predictions[man_range[man_type][0]:man_range[man_type][1]].astype(int)
    
    colors = [cmap(i) for i in range(clusterNum)]
        
    for i in range(clusterNum):
        color_dict.update({i:colors[i]})
        
    color_array_true = [color_dict[i] for i in true_states]
    color_array_predicted = [color_dict[i] for i in predicted_states]
    
    for i, feature in enumerate(features_to_plot):    
        axs[i].grid()
        axs[i].scatter(range(len(df_clus)), df_clus[feature], facecolors='none', edgecolors=color_array_true, 
                       linewidth=2*LINE_WIDTH, s = 5*MARKER_SIZE)
        axs[i].scatter(range(len(df_clus)), df_clus[feature], c=color_array_predicted, s = 5*MARKER_SIZE, 
                       marker = 'o')
        axs[i].set_ylabel(feature+ ' (Nm)', fontsize=25)
        axs[i].tick_params(direction='out', labelsize = 25)
        axs[i].set_xlim((0, len(df_clus)))

        if zoom:        
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            # create the zoomed in version
            axins = zoomed_inset_axes(axs[i], 3, loc=1) # zoom-factor: 3, location: upper-left
            axins.scatter(range(len(df_clus)), df_clus[feature], facecolors='none', edgecolors=color_array_true, 
                           linewidth=5, s = 20*MARKER_SIZE)
            axins.scatter(range(len(df_clus)), df_clus[feature], c=color_array_predicted, s = 10*MARKER_SIZE, 
                           marker = 'o')
            x1, x2, y1, y2 = 25, 55, -1, 1 # specify the limits
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        # dummy scatter for labels
        axs[i].scatter([], [], edgecolors=colors[0], label='Recovery (True)', facecolors='none',
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        axs[i].scatter([], [], edgecolors=colors[1], label='Right-assist (True)', facecolors='none', 
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        axs[i].scatter([], [], edgecolors=colors[2], label='Straight-assist (True)', facecolors='none', 
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        axs[i].scatter([], [], edgecolors=colors[3], label='Release (True)', facecolors='none', 
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        axs[i].scatter([], [], edgecolors=colors[4], label='Braking (True)', facecolors='none', 
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        axs[i].scatter([], [], edgecolors=colors[5], label='Left-assist (True)', facecolors='none', 
                       linewidth=LINE_WIDTH, s = 20*MARKER_SIZE,)
        
        axs[i].scatter([], [], c=np.array(colors[0]).reshape(1,-1), marker = 'o', label='Recovery (Predicted)', s = 10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[1]).reshape(1,-1), marker = 'o', label='Right-assist (Predicted)', s = 10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[2]).reshape(1,-1), marker = 'o', label='Straight-assist (Predicted)', s = 10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[3]).reshape(1,-1), marker = 'o', label='Release (Predicted)', s = 10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[4]).reshape(1,-1), marker = 'o', label='Braking (Predicted)', s = 10*MARKER_SIZE)
        axs[i].scatter([], [], c=np.array(colors[5]).reshape(1,-1), marker = 'o', label='Left-assist (Predicted)', s = 10*MARKER_SIZE)
        
    axs[i].legend(ncol=2, fontsize = 10)
    
    plt.show()
    
    # save confusion matrix
    fig_name = man_type + '_Classification_Clustering.jpg'
    fig_name = os.path.join(path_, fig_name)
    fig.savefig(fig_name)


# In[ ]:


features_to_plot= ['Mean Torque_L', 'Mean Torque_R']
plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type = 'StraightF')


# In[ ]:


features_to_plot= ['Mean Torque_L', 'Mean Torque_R']
plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type = 'Turn90FR')


# In[ ]:


features_to_plot= ['Mean Torque_L', 'Mean Torque_R']
plt_ts_cluster_prediction(test_df, y_pred, int(n_components), features_to_plot, man_type = 'Turn90FL')


# In[ ]:


# detemine label probabilities
labels_prob = model_.predict_proba(X_test)


# In[ ]:


# examine label probabilities
color_ = [cmap(i) for i in range(int(n_components))]

plt.figure(figsize = (18, 10))

for n in range(int(n_components)):
    test_df['Prob_L'+str(n)] = labels_prob[:,n]
    
for i in range(int(n_components)):
    plt.plot(test_df['Prob_L'+str(i)], label = str(i), c = color_[i])

plt.legend()
plt.show()


# In[ ]:


# save labeled dataset
filename = 'Labeled_classification.csv'
filename = os.path.join(path_, filename)
test_df.to_csv(filename)


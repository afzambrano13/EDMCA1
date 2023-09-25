# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:33:40 2023

@author: azamb

Adding Optimal Threshold
"""

import pandas as pd

#Importing data
df=pd.read_csv('ca1-dataset.csv')
df=df.sort_values(by='namea')

#Replacing Target by 0s and 1s
df['OffTask'] = df['OffTask'].replace({'Y': 1, 'N': 0})

#Distribution of each Class (OnTask vs OffTask)
OffTaskSamples=sum(df['OffTask'])/len(df)
print("Off-Task: " + str(OffTaskSamples*100)+ "%")
print("On-Task: " + str(100-OffTaskSamples*100)+ "%")

#Droping columns without variance
columns_to_drop=['Avghelp','Avgchoice','Avgstring','Avgnumber','Avgpoint','Avghelppct-up',
                 'Avgrecent8help','AvgasymptoteA-up','AvgasymptoteB-up']
df = df.drop(columns=columns_to_drop)

#Verifying correlations between features

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Droping Avgnotright because is providing the same information than Avgright
columns_to_drop2=['Avgnotright']  #Other options to drop are Avgprev5Count-up, Avgtimelast5SDnormed and Avgtime
df = df.drop(columns=columns_to_drop2)

#Getting number of unique students
Students=df['namea'].unique()
print('Students: '+str(len(Students)))

#Getting students that were OffTask at some point
OffTaskStudents=df[df['OffTask']==1]['namea'].unique()
print('Off Task Students: '+str(len(OffTaskStudents)))

#Create a column indicating if the student was offtask at any moment (for stratified k fold)
df['AnyOffTask']=0
df.loc[df['namea'].isin(OffTaskStudents), 'AnyOffTask'] = 1

#Groups for student level CV
groups=df['namea']

# Create a 6-fold student level cross-validation object, stratified by the target
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

kf = StratifiedGroupKFold(n_splits=10)
kf2 = StratifiedGroupKFold(n_splits=4)
#Vector for stratified kFold (Not the target!!!)
yKFold = np.array(df['AnyOffTask'])
#Target
y=np.array(df['OffTask'])
#Features
X=np.array(df.drop(columns=['Unique-id','namea','OffTask','AnyOffTask'], axis=1))


#Cross Validation testing several models
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score

#Lists to save results
Kappa_scores=[]
AUC_scores=[]
Precision = []
Recall = []
AllOptimalThresholds=[]

#Flag to activate SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.25,random_state=42)
DoSMOTE=0

#Flag to balance based on class weights
ClassWeights=0

#Cross Validation
for train_index, test_index in kf.split(X, yKFold, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    yKFold_train =  yKFold[train_index]
    groups_train = groups[train_index]
    
    Thresholds=[]
    #Cross Validation for selecting optimal threshold
    for train_index_val, val_index in kf2.split(X_train, yKFold_train, groups_train):
        
        X_trainval, X_val = X[train_index_val], X[val_index]
        y_trainval, y_val = y[train_index_val], y[val_index]

    
        if DoSMOTE==1:
            X_trainval,y_trainval=smote.fit_resample(X_trainval,y_trainval)
        
        if ClassWeights==1:
            # Compute class weights
            class_counts = np.bincount(y)
            total_samples = len(y)
            class_weights = total_samples / (len(class_counts) * class_counts)
            #clf = xgb.XGBClassifier(learning_rate=0.5, n_estimators=200, random_state=5,scale_pos_weight= class_weights[1])
            #clf = LogisticRegression(max_iter=10000, class_weight='balanced')
            #clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=5)
        else:
            #clf = DecisionTreeClassifier()
            #clf = GaussianNB()
            clf = xgb.XGBClassifier(learning_rate=0.5, n_estimators=200, random_state=5)
            #clf = LogisticRegression(max_iter=10000)
            #clf = RandomForestClassifier(n_estimators=200 , random_state=5)
        clf.fit(X_trainval, y_trainval)
        y_valprob=clf.predict_proba(X_val)[:,1]
        
        # Assuming you have true labels y_true and predicted probabilities y_prob
        precision, recall, thresholds = precision_recall_curve(y_val, y_valprob)
        
        # Calculate the F1-score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        # Find the threshold that maximizes the F1-score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        # Add optimal threshold of this iteration
        Thresholds.append(optimal_threshold)
    
    #Mean of optimal thresholds is defined as best threshold
    Best_threshold=np.mean(Thresholds)
    AllOptimalThresholds.append(Best_threshold)
    clf.fit(X_train, y_train)
    
    #Prediction
    y_pred = clf.predict_proba(X_test)[:,1]
    #Binary prediction
    y_pred2 = (y_pred>= Best_threshold).astype(int)
    # Calculate metrics
    AUC = roc_auc_score(y_test, y_pred)
    AUC_scores.append(AUC)
    kappa = cohen_kappa_score(y_test, y_pred2)
    Kappa_scores.append(kappa)
    precision = precision_score(y_test, y_pred2)
    Precision.append(precision)
    recall = recall_score(y_test, y_pred2)
    Recall.append(recall)

# Print the AUC score for each fold
for fold, score in enumerate(AUC_scores, start=1):
    print(f"Fold {fold}: AUC {score:.4f}")

# Calculate and print the mean AUC score across all folds
mean_AUC = np.mean(AUC_scores)
std_AUC = np.std(AUC_scores)
print(f"Mean AUCROC: {mean_AUC:.4f}")
print(f"STD AUCROC: {std_AUC:.4f}\n")

# Print the Kappa for each fold
for fold, score in enumerate(Kappa_scores, start=1):
    print(f"Fold {fold}: Kappa {score:.4f}")

# Calculate and print the mean Kappa across all folds
mean_Kappa = np.mean(Kappa_scores)
print(f"Mean Kappa: {mean_Kappa:.4f}")
std_Kappa = np.std(Kappa_scores)
print(f"STD Kappa: {std_Kappa:.4f}\n")

# Print the Precision for each fold
for fold, score in enumerate(Precision, start=1):
    print(f"Fold {fold}: Precision {score:.4f}")

# Calculate and print the mean Precision across all folds
mean_Precision = np.mean(Precision)
print(f"Mean Precision: {mean_Precision:.4f}")
std_Precision = np.std(Precision)
print(f"STD Precision: {std_Precision:.4f}\n")

# Print the Recall for each fold
for fold, score in enumerate(Recall, start=1):
    print(f"Fold {fold}: Recall {score:.4f}")

# Calculate and print the mean Recall across all folds
mean_Recall = np.mean(Recall)
print(f"Mean Recall: {mean_Recall:.4f}")
std_Recall = np.std(Recall)
print(f"STD Recall: {std_Recall:.4f}\n")

# Calculate and print the mean Recall across all folds
mean_Threshold = np.mean(AllOptimalThresholds)
print(f"Mean Threshold: {mean_Threshold:.4f}")
std_Threshold = np.std(AllOptimalThresholds)
print(f"STD Threshold: {std_Threshold:.4f}")
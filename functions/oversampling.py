import random
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt


# Different functions for different oversampling techniques
def SMOTE_dataset(thresh, X_train, Y_train):
    Labels=pd.DataFrame(Y_train)
    class_size = Labels.groupby('Class').size()

    sampling_strategy={}
    for i in range(len(class_size)):
        if(class_size[i] < thresh):
            sampling_strategy[i]=thresh
            
    smt=SMOTE(sampling_strategy={0:2000, 1:2000, 3:2000, 6:2000})
    X,Y=smt.fit_resample(X_train, Y_train)
    
    train_data=np.array(X)
    train_labels=np.array(Y)
    x_train, y_train = torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long()
    
    return X_train, Y_train, x_train, y_train

def SMOTEENN_dataset(thresh, X_train, Y_train):
    Labels=pd.DataFrame(Y_train)
    class_size = Labels.groupby('Class').size()

    sampling_strategy={}
    for i in range(len(class_size)):
        if(class_size[i] < thresh):
            sampling_strategy[i]=thresh
            
    smt=SMOTE(sampling_strategy={0:2000, 1:2000, 3:2000, 6:2000})
    X,Y=smt.fit_resample(X_train, Y_train)
    
    train_data=np.array(X)
    train_labels=np.array(Y)
    x_train, y_train = torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long()
    
    return X_train, Y_train, x_train, y_train

def ADAYSN(thresh, X_train, Y_train):
    Labels=pd.DataFrame(Y_train)
    class_size = Labels.groupby('Class').size()

    sampling_strategy={}
    for i in range(len(class_size)):
        if(class_size[i] < thresh):
            sampling_strategy[i]=thresh
            
    smt=SMOTE(sampling_strategy={0:2000, 1:2000, 3:2000, 6:2000})
    X,Y=smt.fit_resample(X_train, Y_train)
    
    train_data=np.array(X)
    train_labels=np.array(Y)
    x_train, y_train = torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long()
    
    return X_train, Y_train, x_train, y_train

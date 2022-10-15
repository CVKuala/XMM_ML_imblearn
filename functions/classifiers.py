import random
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Functions for standardization and classification
def Standardize(data):
    
    scaler=preprocessing.StandardScaler()
    length_data=len(data)
    data.index=np.array(range(length_data))
    main_data=data[data.columns[0:-1]]
    labels=data[data.columns[-1]]
    columns=main_data.columns

    main_data=scaler.fit_transform(main_data)
    return main_data, labels, columns

def RF_Classification(X_train, Y_train, X_test, Y_test):
    
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, Y_train)
    y_pred_rf1 = clf.predict(X_test)
    print("ACCURACY OF THE Random_Forest_Classifier: ", metrics.accuracy_score(Y_test, y_pred_rf1))
    
def XGB_Classification(X_train, Y_train, X_test, Y_test):
    
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, Y_train)
    y_pred_xgb1 = xgb_clf.predict(X_test)
    print("ACCURACY OF THE XGB_Classifier: ", metrics.accuracy_score(Y_test, y_pred_xgb1))

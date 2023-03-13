### IMPORTS

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as Sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import time
import statistics as st
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
#from tabgan.sampler import GANGenerator
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../')


#######

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type = str, default = "Glass0", help = "Dataset", \
                        choices = ["Glass0", "Glass1", "Glass2", "Glass5", "Glass6", "Glass123vs567", 
                        "Glass5vs12", "Glass016vs5", "Yeast1", "Yeast3", "Yeast4", "Yeast5", "Yeast6", 
                        "Yeast2vs8", "Yeast1vs7", "Yeast1289vs7", "Yeast1458vs7", "Ecoli0vs1", "Ecoli1", 
                        "Ecoli2", "Ecoli3", "Ecoli4", "Ecoli0137vs26"])
    
    parser.add_argument("--augmentation", type = str, default = "No Aug", help = "Data Augmentation Technique", \
                        choices = ["No Aug", "SMOTE", "SMOTE-ENN", "Borderline SMOTE", "SMOTE-SVM", "KMeans SMOTE", 
                                   "ADASYN", "ROS", "RUS", "CT-GAN"])
    
    parser.add_argument("--ensemble", type = str, default = "Adaboost", help = "Model", \
                        choices = ["Adaboost", "XGBoost", "GradientBoost", "LGBM", "Decision Tree", "Random Forest", 
                                   "Voting Classifier 1", "Voting Classifier 2", "Stacking Classifier 1", "Stacking Classifier 2"])
    

    args = parser.parse_args()

    if args.data == "Glass0":
      df = pd.read_csv("/Documents/Glass/Glass0")
    if args.data == "Glass1":
      df = pd.read_csv("/Documents/Glass/Glass1")
    if args.data == "Glass2":
      df = pd.read_csv("/Documents/Glass/Glass2")
    if args.data == "Glass5":
      df = pd.read_csv("/Documents/Glass/Glass5")
    if args.data == "Glass6":
      df = pd.read_csv("/Documents/Glass/Glass6")
    if args.data == "Glass123vs567":
      df = pd.read_csv("/Documents/Glass/Glass123vs567")
    if args.data == "Glass5vs12":
      df = pd.read_csv("/Documents/Glass/Glass5vs12")
    if args.data == "Glass016vs5":
      df = pd.read_csv("/Documents/Glass/Glass016vs5")
    if args.data == "Yeast1":
      df = pd.read_csv("/Documents/Yeast/Yeast1")
    if args.data == "Yeast3":
      df = pd.read_csv("/Documents/Yeast/Yeast3")
    if args.data == "Yeast4":
      df = pd.read_csv("/Documents/Yeast/Yeast4")
    if args.data == "Yeast5":
      df = pd.read_csv("/Documents/Yeast/Yeast5")
    if args.data == "Yeast6":
      df = pd.read_csv("/Documents/Yeast/Yeast6")
    if args.data == "Yeast2vs8":
      df = pd.read_csv("/Documents/Yeast/Yeast2vs8")
    if args.data == "Yeast1vs7":
      df = pd.read_csv("/Documents/Yeast/Yeast1vs7")
    if args.data == "Yeast1289vs7":
      df = pd.read_csv("/Documents/Yeast/Yeast1289vs7")      
    if args.data == "Yeast1458vs7":
      df = pd.read_csv("/Documents/Yeast/Yeast1458vs7")
    if args.data == "Ecoli0vs1":
      df = pd.read_csv("/Documents/Ecoli/Ecoli0vs1")
    if args.data == "Ecoli1":
      df = pd.read_csv("/Documents/Ecoli/Ecoli1")
    if args.data == "Ecoli2":
      df = pd.read_csv("/Documents/Ecoli/Ecoli2")
    if args.data == "Ecoli3":
      df = pd.read_csv("/Documents/Ecoli/Ecoli3")
    if args.data == "Ecoli4":
      df = pd.read_csv("/Documents/Ecoli/Ecoli4")
    if args.data == "Ecoli0137vs26":
      df = pd.read_csv("/Documents/Ecoli/Ecoli0137vs26")

    y = df['label']
    X = df.drop(['label'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    if args.augmentation == "SMOTE":
      sm = SMOTE()
    if args.augmentation == "SMOTE-ENN":
      sm = SMOTEENN()
    if args.augmentation == "Borderline SMOTE":
      sm = BorderlineSMOTE(kind = 'borderline-1') 
    if args.augmentation == "SMOTE-SVM":
      sm = SVMSMOTE()
    if args.augmentation == "KMeans SMOTE":
      sm = KMeansSMOTE(sampling_strategy='auto')
    if args.augmentation == "ADASYN":
      sm = ADASYN() 
    if args.augmentation == "RUS":
      sm = RandomUnderSampler() 
    if args.augmentation == "ROS":
      sm = RandomOverSampler()

    if args.ensemble == "Adaboost":
      classi = AdaBoostClassifier()
    if args.ensemble == "XGBoost":
      classi = xgboost.XGBClassifier()
    if args.ensemble == "GradientBoost":
      classi = GradientBoostingClassifier()
    if args.ensemble == "LGBM":
      classi = lightgbm.LGBMClassifier()
    if args.ensemble == "Decision Tree":
      classi = DecisionTreeClassifier()
    if args.ensemble == "Random Forest":
      classi = RandomForestClassifier()
    if args.ensemble == "Voting Classifier 1":
      dtc =  DecisionTreeClassifier()
      rfc = RandomForestClassifier()
      knn =  KNeighborsClassifier()
      xgb = xgboost.XGBClassifier()
      classi = VotingClassifier(estimators=[('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)], voting='soft')
    if args.ensemble == "Voting Classifier 2":
      rfc = RandomForestClassifier()
      xgb = xgboost.XGBClassifier()
      classi = VotingClassifier(estimators=[('rfc',rfc),('xgb',xgb)], voting='soft')
    if args.ensemble == "Stacking Classifier 1":
      dtc =  DecisionTreeClassifier()
      rfc = RandomForestClassifier()
      knn =  KNeighborsClassifier()
      xgb = xgboost.XGBClassifier()
      clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] 
      lr = LogisticRegression()
      classi = StackingClassifier( estimators = clf,final_estimator = lr)
    if args.ensemble == "Stacking Classifier 2":
      rfc = RandomForestClassifier()
      xgb = xgboost.XGBClassifier()
      clf = [('rfc',rfc),('xgb',xgb)] 
      lr = LogisticRegression()
      classifier = StackingClassifier( estimators = clf,final_estimator = lr)



    if args.augmentation == "No Aug":
      acc_list = []
      f1_list = []
      roc_list = []
      for i in range(30):
        classi.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_pred))

    if args.augmentation != "No Aug":
      acc_list = []
      f1_list = []
      roc_list = []
      for i in range(30):
        X_train, y_train = sm.fit_resample(X_train, y_train)
        classi.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_pred))
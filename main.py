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
import pandas as pd
import io
import os

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
                                   "VCI", "VCII", "SCI", "SCII"])
    

    args = parser.parse_args()

    if args.data == "Glass0":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass0.csv")
    if args.data == "Glass1":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass1.csv")
    if args.data == "Glass2":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass2.csv")
    if args.data == "Glass5":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass5.csv")
    if args.data == "Glass6":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass6.csv")
    if args.data == "Glass123vs567":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass0123vs567.csv")
    if args.data == "Glass5vs12":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass5vs12.csv")
    if args.data == "Glass016vs5":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Glass%20Datasets/Glass016vs5.csv")
    if args.data == "Yeast1":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast1.csv")
    if args.data == "Yeast3":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast3.csv")
    if args.data == "Yeast4":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast4.csv")
    if args.data == "Yeast5":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast5.csv")
    if args.data == "Yeast6":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast6.csv")
    if args.data == "Yeast2vs8":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast2vs8.csv")
    if args.data == "Yeast1vs7":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast1vs7.csv")
    if args.data == "Yeast1289vs7":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast1289vs7.csv")      
    if args.data == "Yeast1458vs7":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Yeast%20Datasets/Yeast1458vs7.csv")
    if args.data == "Ecoli0vs1":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli0vs1.csv")
    if args.data == "Ecoli1":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli1.csv")
    if args.data == "Ecoli2":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli2.csv")
    if args.data == "Ecoli3":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli3.csv")
    if args.data == "Ecoli4":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli4.csv")
    if args.data == "Ecoli0137vs26":
      df = pd.read_csv("https://raw.githubusercontent.com/sydney-machine-learning/review-classimbalanced-ensemblelearning/main/Datasets/Ecoli%20Datasets/Ecoli0137vs26.csv")

    df = df.drop(['Unnamed: 0'], axis = 1)

    lencoders = {}
    for col in df.select_dtypes(include=['object']).columns:
      lencoders[col] = LabelEncoder()
      df[col] = lencoders[col].fit_transform(df[col])

    y = df['label']
    X = df.drop(['label'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)

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
    if args.ensemble == "VCI":
      dtc =  DecisionTreeClassifier()
      rfc = RandomForestClassifier()
      knn =  KNeighborsClassifier()
      xgb = xgboost.XGBClassifier()
      classi = VotingClassifier(estimators=[('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)], voting='soft')
    if args.ensemble == "VCII":
      rfc = RandomForestClassifier()
      xgb = xgboost.XGBClassifier()
      classi = VotingClassifier(estimators=[('rfc',rfc),('xgb',xgb)], voting='soft')
    if args.ensemble == "SCI":
      dtc =  DecisionTreeClassifier()
      rfc = RandomForestClassifier()
      knn =  KNeighborsClassifier()
      xgb = xgboost.XGBClassifier()
      clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] 
      lr = LogisticRegression()
      classi = StackingClassifier(estimators = clf,final_estimator = lr)
    if args.ensemble == "SCII":
      rfc = RandomForestClassifier()
      xgb = xgboost.XGBClassifier()
      clf = [('rfc',rfc),('xgb',xgb)] 
      lr = LogisticRegression()
      classi = StackingClassifier(estimators = clf,final_estimator = lr)



    if args.augmentation == "No Aug":
      acc_list = []
      f1_list = []
      roc_list = []
      for i in range(30):
        classi.fit(X_train, y_train)
        y_pred = classi.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_pred))
      print("Accuracy - ", st.mean(acc_list), "(", max(acc_list), ",", st.stdev(acc_list),")")
      print("F1-score - ", st.mean(f1_list), "(", max(f1_list), ",", st.stdev(f1_list),")")
      print("ROC - ", st.mean(roc_list), "(", max(roc_list), ",", st.stdev(roc_list),")")

    if args.augmentation != "No Aug":
      acc_list = []
      f1_list = []
      roc_list = []
      for i in range(30):
        X_train, y_train = sm.fit_resample(X_train, y_train)
        classi.fit(X_train, y_train)
        y_pred = classi.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_pred))
      
      print("Accuracy - ", st.mean(acc_list), "(", max(acc_list), ",", st.stdev(acc_list),")")
      print("F1-score - ", st.mean(f1_list), "(", max(f1_list), ",", st.stdev(f1_list),")")
      print("ROC - ", st.mean(roc_list), "(", max(roc_list), ",", st.stdev(roc_list),")")
# build the machine learning model and saved as a pickle file
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# Check the model
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# import the data (this contains only relevant features from out previous analysis in jupyter notebook)
df = pd.read_csv('new_HR.csv')
print(len(df.columns))
print(df.columns)

# Encode Categorical variables
dummies = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
df['OverTime'] = df['OverTime'].replace({'Yes': 1, 'No': 0})
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
for var in dummies:
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
df.drop(labels=dummies, axis=1, inplace=True)
print(len(df.columns))
y = df['Attrition']
X = df.drop('Attrition', axis=1)

# Feature Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Apply SMOTE
sm = SMOTE(random_state=0)
X_os,y_os=sm.fit_resample(X,y)

# Split the data - for checking model
# Xos_train, Xos_test, yos_train, yos_test = train_test_split(X_os, y_os, test_size=0.25, random_state=42)

# Train & fit the XGB Model
XGB_model = XGBClassifier()
# XGB_model.fit(Xos_train, yos_train) # for checking model
XGB_model.fit(X_os, y_os)

# Make Predictions & check accuracy
# XGB_pred = XGB_model.predict(Xos_test)
# print(classification_report(yos_test, XGB_pred))

import pickle
# save trained model to disk
pickle.dump(XGB_model, open('XGB_model.pkl', 'wb'))
# XGB_model=pickle.load(open('XGB_model.pkl', 'rb'))


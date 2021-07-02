# apply the trained model to predict the class label by using input parameters from the sidebar panel of the web app’s front-end.
# This file will serve the web app that will allow predictions to be made using the machine learning model loaded from the
# pickled file. The web app accepts inout values from 2 sources:
# Feature values from the slider bars.
# Feature values from the uploaded CSV file.

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

st.write("""
# HR Attrition Prediction App
This app predicts which employees are more likely to stay or quit! 

Data obtained from https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
""")

# st.subheader('Example CSV Input File')
# df=pd.read_csv('new_HR.csv') # upload the data chossen above
# st.success("Data Sucessfully loaded")
# st.dataframe(df.head())

st.subheader('Input Feature Options')
st.info('***Education:***  \n'  '1: Below College  \n' '2: College  \n'  '3: Bachelor  \n' '4: Master  \n' '5: Doctor  \n')
st.info('***Environment Satisfaction:***  \n'  '1: Low  \n' '2: Medium  \n'  '3: High  \n' '4: Very High')
st.info('***Job Involvement:***  \n'  '1: Low  \n' '2: Medium  \n'  '3: High  \n' '4: Very High')
st.info('***Job Satisfaction:***  \n'  '1: Poor  \n' '2: Good  \n'  '3: Excellent  \n' '4: Outstanding')
st.info('***Relationship Satisfaction:***  \n'  '1: Low  \n' '2: Medium  \n'  '3: High  \n' '4: Very High')
st.info('***Work Life Balance (time spent between work and outside):***  \n'  '1: Bad  \n' '2: Good  \n'  '3: Better  \n' '4: Best')
st.info('***Stock Option Level:***  \n'  'Enter 0,1,2 or 3')





# Collects user input features into dataframe
st.sidebar.header('User Input Features')

def user_input_features():
    Age = st.sidebar.slider('Age',16, 70)
    BusinessTravel = st.sidebar.selectbox('Business Travel', ('Non-Travel', 'Travel_Frequently', 'Travel_Rarely'))
    Department= st.sidebar.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
    DistanceFromHome = st.sidebar.slider('Distance (km)', 1, 50)
    Education = st.sidebar.slider('Education', 1,5)
    EducationField = st.sidebar.selectbox('Education Field', ('Life Sciences', 'Medical','Marketing', 'Technical Degree', 'Human Resources', 'Other'))
    EnvironmentSatisfaction = st.sidebar.slider('Environment Satisfaction', 1,4)
    JobInvolvement = st.sidebar.slider('Job Involvement', 1, 4)
    JobRole = st.sidebar.selectbox('Job Role', ('Sales Executive', 'Research Scientist','Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                                'Sales Representative', 'Research Director', 'Human Resources'))
    JobSatisfaction = st.sidebar.slider('Job Satisfaction', 1, 4)
    MaritalStatus = st.sidebar.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    MonthlyIncome = st.sidebar.slider('Monthly Income', 1000, 20000)
    NumCompaniesWorked = st.sidebar.slider('Num Companies Worked', 1, 10)
    OverTime = st.sidebar.selectbox('Over Time', ('Yes', 'No'))
    RelationshipSatisfaction = st.sidebar.slider('Relationship Satisfaction', 1, 4)
    StockOptionLevel = st.sidebar.slider('Stock Option Level', 0, 3)
    TotalWorkingYears = st.sidebar.slider('Total Working Years', 0, 40)
    WorkLifeBalance = st.sidebar.slider('Work Life Balance', 1, 4)
    YearsInCurrentRole = st.sidebar.slider('Years InCurrent Role', 0, 20)
    YearsSinceLastPromotion = st.sidebar.slider('Years Since Last Promotion', 0, 20)
    YearsWithCurrManager = st.sidebar.slider('Years With Current Manager', 0, 20)


    data ={'Age': Age, 'BusinessTravel': BusinessTravel, 'Department': Department, 'DistanceFromHome':DistanceFromHome,'Education':Education,'EducationField':EducationField,
           'EnvironmentSatisfaction':EnvironmentSatisfaction,'JobInvolvement':JobInvolvement,'JobRole':JobRole,'JobSatisfaction':JobSatisfaction,
           'MaritalStatus':MaritalStatus,'MonthlyIncome':MonthlyIncome,'NumCompaniesWorked':NumCompaniesWorked,'OverTime':OverTime,'RelationshipSatisfaction':RelationshipSatisfaction,
           'StockOptionLevel':StockOptionLevel,'TotalWorkingYears':TotalWorkingYears,'WorkLifeBalance':WorkLifeBalance,'YearsInCurrentRole':YearsInCurrentRole,
           'YearsSinceLastPromotion':YearsSinceLastPromotion,'YearsWithCurrManager':YearsWithCurrManager}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.success("Data Sucessfully loaded")
st.write(input_df)

# Combines user input features with entire HR dataset
# This will be useful for the encoding phase
hr_raw = pd.read_csv('new_HR.csv')
# hr_raw['Attrition'] = hr_raw['Attrition'].map({'Yes':1, 'No':0})

# y = hr_raw['Attrition']
hr_raw = hr_raw.drop('Attrition', axis=1)
df = pd.concat([input_df,hr_raw],axis=0)

# Encode Categorical variables
df['OverTime'] = df['OverTime'].replace({'Yes': 1, 'No': 0})
dummies = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
for var in dummies:
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
df.drop(labels=dummies, axis=1, inplace=True)
# st.write(df)

# Feature Scaling
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
# st.write(df)

# select the top row (user input)
df = df[:1] # Selects only the first row (the user input data)
# st.write(df)

# # Reads in saved classification model
load_XGB = pickle.load(open('XGB_model.pkl', 'rb'))

# Apply model to make predictions
# Employee leaving the company (0=no, 1=yes)
prediction = load_XGB.predict(df)
prediction_proba = load_XGB.predict_proba(df)

if prediction == 1:
    prediction = 'Yes'
else:
    prediction = 'No'
st.subheader('Prediction: Employee leaving the company (0=No, 1=Yes)')
st.write('Prediction = ',prediction)

st.write('Prediction Probability')
st.write(prediction_proba)
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:43:26 2020

@author: Iacopo
"""

#Import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
df = pd.read_excel('marketing_campaign.xlsx')
df.info

#Missing values 
df.isna().sum() #searchig for missings
from sklearn.preprocessing import Imputer #Importing Imputer module to handle missing values
imputer = Imputer()
imputer = imputer.fit(df.loc[:,['Income']])
df['Income'] = imputer.transform(df.loc[:,['Income']])

#Data Explorationdf
df['Age'] = 2020-df['Year_Birth'] #Feature engineering of Age variable to be represented in a distplot
age_dist= sns.distplot(df['Age'], kde = True) #Distribution of age
education_count = sns.countplot(x='Education',data=df) #Education level distribution
marstatus_count = sns.countplot(x='Marital_Status',data=df) #Marital status distribution
income_dist = sns.distplot(df['Income'],kde=True) #Income distribution
data = df #Rename the dataset to enable the application of the map function without modifying original values of df 
data['Customers accepting campaign 1'] = data['AcceptedCmp1'].map({0: 'no', 1: 'yes'}) #Mapping new values 0 --> no, 1 --> yes
accepted_campaign1 = sns.countplot(data['Customers accepting campaign 1']) #Plotting clients accepting/not accepting camp1
data['Customers accepting campaign 2'] = data['AcceptedCmp2'].map({0: 'no', 1: 'yes'}) 
accepted_campaign2 = sns.countplot(data['Customers accepting campaign 2']) #Camp2
data['Customers accepting campaign 3'] = data['AcceptedCmp3'].map({0: 'no', 1: 'yes'})
accepted_campaign3 = sns.countplot(data['Customers accepting campaign 3']) #Camp3
data['Customers accepting campaign 4'] = data['AcceptedCmp4'].map({0: 'no', 1: 'yes'})
accepted_campaign4 = sns.countplot(data['Customers accepting campaign 4']) #Camp 4
data['Customers accepting campaign 5'] = data['AcceptedCmp5'].map({0: 'no', 1: 'yes'})
accepted_campaign5 = sns.countplot(data['Customers accepting campaign 5'])#Camp5
data['Complain'] = data['Complain'].map({0: 'no', 1: 'yes'})
complain = sns.countplot(data['Complain'])
data['Response current campaign'] = data['Response'].map({0: 'no', 1: 'yes'})
response = sns.countplot(data['Response current campaign']) #Plotting responses to current campaign
wine = sns.distplot(df['MntWines'],hist=False) #Distribution of the expended amount for wines
fruit = sns.distplot(df['MntFruits'],hist=False) #Distribution of the expended amount for fruit
meat = sns.distplot(df['MntMeatProducts'],hist=False) #Distribution of the expended amount for meat
fish = sns.distplot(df['MntFishProducts'],hist=False) #Distribution of the expended amount for fish
sweet = sns.distplot(df['MntSweetProducts'],hist=False) #Distribution of the expended amount for sweet
gold = sns.distplot(df['MntGoldProds'],hist=False) #Distribution of the expended amount for gold
store_purch = sns.countplot(x='NumStorePurchases',data=df) #Distribution of number of purchases at the store
deal_purch = sns.countplot(x='NumDealsPurchases',data=df) #Distribution of number of discounted purchases 
web_purch = sns.countplot(x='NumWebPurchases',data=df) #Distribution of number of number web purchases
catalogue_purch = sns.countplot(x='NumCatalogPurchases',data=df) #Distribution of number catalogue purchases
webvis_month = sns.countplot(x='NumWebVisitsMonth',data=df) #Distribution of web visits

#Feature Engineering
df['Education'] = df['Education'].map({'Basic':0,'Graduation':1,'Master':1,'2n Cycle':1, 'PhD':2}) #map function assign a numeric number to each category
df['Marital_Status'] = df['Marital_Status'].map({'Single':0,'Alone':0,'Widow':0,'Divorced':1, 'Absurd':1,'YOLO':1,'Together':2,'Married':2}) #see above

#Standardization
from sklearn.preprocessing import StandardScaler #Importing standard scaler module for feature scaling (standardization)
df[['Income','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumStorePurchases'
   ,'NumWebVisitsMonth','Age']]=StandardScaler().fit_transform(df[['Income','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumStorePurchases'
   ,'NumWebVisitsMonth','Age']])
   
#Import SMOTE library  
df=df.drop(['ID','Z_CostContact','Z_Revenue','Year_Birth','Dt_Customer'],axis=1) #Dropping variable of no interest
from imblearn.over_sampling import SMOTE #Importing SMOTE module to balance the dataset
sm = SMOTE(sampling_strategy='minority', random_state=7) #Resample the minority class
X, y = sm.fit_sample(df.drop('Response', axis=1), df['Response']) #Fit the model to generate the data, and create X and Y
X = pd.DataFrame(X)

#Input Selection
from sklearn.feature_selection import RFE #Importing Recursive Feature Elimination module for input selection
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=12, step=10, verbose=5) #Creating input selector with LR as estimator
rfe_selector.fit(X, y) #Fitting the selector on X and y
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist() #Determining most relevant features
X = X.iloc[:,[0,1,4,5,8,15,16,17,18,19,20,21]]

#Training Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0) #Splitting data into training and test

##SVM
from sklearn.svm import SVC #Import SVC to run SVM
svm = SVC() #Creating svm classifier
svm.fit(X_train,y_train) #Fitting the classifier to the training set

#Predictions 
y_pred = svm.predict(X_test) #Testing the predictions on the test set

#Evaluation
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score
print(confusion_matrix(y_test,y_pred)) #Confusion mattrix
print(classification_report(y_test,y_pred)) #Precision, Recall, Accuracy, F1-score
print(roc_auc_score(y_test, y_pred)) #Area under the curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)') #Plotting roc curve
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Optimization
param_grid = {'C': [0.01,0.1, 10, 100, 1000],
              'gamma': [1,0.1,0.01,0.001,0.0001]} #Creating parameters grid (C and gamma)
from sklearn.model_selection import GridSearchCV #Importing GrdiSearchCV module to apply grid search
grid = GridSearchCV(SVC(), param_grid) #Applying Grid Search to SVC estimator
grid.fit(X_train, y_train) #Fitting the grid object to the training and test set
grid.best_params_ #Searching for best parameters
grid_predictions = grid.predict(X_test) #Testing model prediction performance
print(confusion_matrix(y_test,grid_predictions)) #Confusion Matrix
print(classification_report(y_test,grid_predictions)) #Precision, Recall, f1, accuracy
print(roc_auc_score(y_test, grid_predictions)) #Area under the curve
fpr, tpr, _ = roc_curve(y_test, grid_predictions) #Plotting ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

##DECISION TREE
from sklearn.tree import DecisionTreeClassifier #Importing Decision Tree module
Dtree = DecisionTreeClassifier() #Creating Decision Tree estimator
Dtree.fit(X_train,y_train) #Fitting Decision Tree to the training and test set

#Predictions
y_pred = Dtree.predict(X_test) #Testing model prediction performance on the test set

#Model Evaluation
print(confusion_matrix(y_test,y_pred)) 
print(classification_report(y_test,y_pred))
print(roc_auc_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

##RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier #Importing Random Forest Classifier module
rfc = RandomForestClassifier(max_depth= None, max_features=5, n_estimators=50) 
#Creating rfc estimator (parameters have been obtained after running hyperparameter tuning; see below)
rfc.fit(X_train, y_train) #Fitting rfc to the training and test set
rfc_pred = rfc.predict(X_test) #Testing model prediction performance
print(classification_report(y_test,rfc_pred)) #precision, recall, f1, accuracy
print(confusion_matrix(y_test,rfc_pred)) #confusion matric
print(roc_auc_score(y_test, y_pred)) #Computing area under the curve
fpr, tpr, _ = roc_curve(y_test, y_pred) #Plotting Roc Curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Optimization RF
param_grid = {'n_estimators': [10, 25, 50, 100], 'max_features': [5, 10], 'max_depth': [10, 50, None]} #Creating parameter grid 
from sklearn.model_selection import GridSearchCV #Importing Grid Search CV module
grid = GridSearchCV(RandomForestClassifier(), param_grid) #Creating grid object applying Grid Search to the RFC
grid.fit(X_train, y_train) #Fitting grid to the training set and the test set
grid.best_params_ #Searching for best parameters
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
print(roc_auc_score(y_test, grid_predictions))
fpr, tpr, _ = roc_curve(y_test, grid_predictions)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Features importance
importances = rfc.feature_importances_ #Determining features importance
importances = pd.DataFrame(importances) #Converting float to dataframe
importances['FEATURE'] = ['Education','Marital Status','Teen at home','Recency (N of days since last purchase)','Amnt Spent for Meat Products',
'N of Store Purchases','N of web visits','Accepted campaign 1','Accepted campaign 2','Accepted campaign 3','Accepted campaign 4','Accepted campaign 5'] 
#Creating a column of the dataframe containing features name
importances = importances.rename(columns={0:'SCORE'}) #Renaming column "0"
importances = importances[['FEATURE','SCORE']] #Inverting FEATURE AND SCORE columns
d=importances.sort_values(by = 'SCORE',ascending=False) #Sorting values by score

sns.barplot(x=d['SCORE'],y=d['FEATURE']) #Plotting features based on their scores' imporance
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title('Features model importance')
plt.show()

#RFC Model validation 10-cross validation
from sklearn.model_selection import cross_validate #Importing cross_validate module (it allows to compute multiple metrics)
from sklearn.model_selection import cross_val_score #Importing cross_val_score (it allows to obtain accuracy)
scores = cross_validate(rfc, X, y, cv=10,scoring= ['precision_macro','recall_macro','f1_macro']) #Running 10-fold cross validation (evaluation: precision, recall, f1)
score = cross_val_score(rfc, X, y, cv=10) #Running 10-fold cross validation (accuracy)
iterations = [1,2,3,4,5,6,7,8,9,10] 
recall = scores['test_recall_macro']
precision = scores['test_precision_macro']
f1 = scores['test_f1_macro']
accuracy = score
table = {'iterations':[1,2,3,4,5,6,7,8,9,10],'precision':[0.82657548, 0.82151149, 0.965187, 0.9570132, 
         0.9593744, 0.96940989, 0.96156195, 0.96084546, 0.97263682, 0.97029703],
         'recall':[0.77486911, 0.80366492, 0.96335079, 0.95549738, 0.95811518,
          0.96858639, 0.96052632, 0.96052632, 0.97105263, 0.96842105],
          'f1':[0.76559066, 0.80090202, 0.96331458, 0.95546045, 0.95808646,
           0.9685726, 0.96050416, 0.96051948, 0.97102835, 0.96838953],
            'accuracy':[0.76963351, 0.79319372, 0.95549738, 0.95811518, 0.95811518,
            0.96858639, 0.96842105, 0.96315789, 0.96578947, 0.95789474]}
results = pd.DataFrame(table) #Creating a table summarizing all the metrics

#Plotting cross-validation metrics 
sns.lineplot(x=results['iterations'],y=results['recall'],color='coral')
sns.lineplot(x=results['iterations'],y=results['precision'],color='coral')
sns.lineplot(x=results['iterations'],y=results['f1'],color='coral')
sns.lineplot(x=results['iterations'],y=results['accuracy'],color='coral')


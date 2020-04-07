# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:21:20 2020

@author: Iacopo
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Laod dataset
df = pd.read_csv('bank-additional.csv', sep = ';')

#Exploring Dataset
df.head()
df.info()

#Selecting features
X = df.iloc[:,0:20]
y = df.iloc[:,20]

#Dealing with categorical variables
cols_at_end = ['age','duration','campaign','pdays','previous']
X = X[[c for c in X if c not in cols_at_end] 
        + [c for c in cols_at_end if c in X]]
X1 = X.iloc[:,0:10]
X2 = X.iloc[:,10:]
X1 = pd.get_dummies(X1, drop_first=True)
X = pd.concat([X1,X2], axis = 1)

y = pd.get_dummies(y)
y = y['yes']

#Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)

#Training the model
from sklearn.svm import SVC
svm = SVC(probability=True)
svm.fit(X_train, y_train)

#Predictions
y_pred = svm.predict(X_test)

#Evaluation
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc
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

#Applying k-Fold Cross Validation (k = 10)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 10)

#Model evaluation after K-Fold Cross Validation: Precision, Recall(sensitivity), f1 score, AUC
accuracies.mean()
accuracies.std()
f1= cross_val_score(estimator = svm, X = X_train, y = y_train, cv=10, scoring='f1_weighted')
f1score=round((f1.mean()*100),3)
Pscores= cross_val_score(estimator = svm, X = X_train, y = y_train, cv=10, scoring='precision_weighted')
precision=round((Pscores.mean()*100),3)
recall = cross_val_score(estimator = svm, X = X_train, y = y_train, cv=10, scoring='recall_weighted')
sensitivity=round((recall.mean()*100),3)
y_train1 = 1-y_train #for Specificity
recall = cross_val_score(estimator = svm, X = X_train, y = y_train1, cv=10, scoring='recall_weighted')#for specificity
specificity=round((recall.mean()*100),3)

from sklearn.model_selection import StratifiedKFold
from scipy import interp
cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_train, y_train):
    probas_ = svm.fit(X_train,y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
plt.title('Cross-Validation ROC of SVM',fontsize=10)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()







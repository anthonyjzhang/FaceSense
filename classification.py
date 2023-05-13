import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
 
# Display csv data
data_raw = pd.read_csv('clf_raw.csv',header=None)
data_raw.rename(columns={data_raw.columns[-1]:'label'}, inplace=True)
print(data_raw)

def cvs(X,y,clf):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_list = []
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # give me accuracy, precision, recall, f1-score
        
        accuracy_list.append(metrics.accuracy_score(y_test, y_pred))
    
    # return the average accuracy
    return accuracy_list

# Split the data into train and test data
X = data_raw.iloc[:,:-1]
y = data_raw.iloc[:,-1]

# Use KNN; K = 10
y = y.apply(lambda x: 0 if x==0 else 1)
clf = KNeighborsClassifier(n_neighbors=10)
scores = cvs(X,y,clf)
print(scores)

i = 'clf_pca100%.csv'
num = re.findall(r"\d+\.?\d*",i)
precent = num[0]
data_pca = pd.read_csv(i,header=None)
data_pca.rename(columns={data_pca.columns[-1]:'label'}, inplace=True)
X = data_pca.iloc[:180,:-1]
y = data_pca.iloc[:180,-1]
y = y.apply(lambda x: 0 if x==0 else 1)
clf = KNeighborsClassifier(n_neighbors= 10)
scores = cvs( X, y,clf)
print('Current PCA precentage is: ', precent)
print('The mean accuracy score is: ', scores)
print(np.mean(scores))
print('-'*50)

# BINARY CLASSIFICATION

# Binary classification using KNN
filenames = ['clf_pca61%.csv','clf_pca70%.csv','clf_pca80%.csv','clf_pca85%.csv','clf_pca90%.csv','clf_pca100%.csv']
acc = []
for i in filenames:
    num = re.findall(r"\d+\.?\d*",i)
    precent = num[0]
    data_pca = pd.read_csv(i,header=None)
    data_pca.rename(columns={data_pca.columns[-1]:'label'}, inplace=True)
    X = data_pca.iloc[:180,:-1]
    y = data_pca.iloc[:180,-1]
    y = y.apply(lambda x: 0 if x==0 else 1)
    clf = KNeighborsClassifier(n_neighbors=10)
    scores = cvs( X, y,clf)
    print('Current PCA precentage is: ', precent)
    print('The mean accuracy score is: ', np.mean(scores))
    print('-'*50)
    acc.append(np.mean(scores))

# Binary classification using Random Forests
filenames = ['clf_pca61%.csv','clf_pca70%.csv','clf_pca80%.csv','clf_pca85%.csv','clf_pca90%.csv','clf_pca100%.csv']
acc = []
for i in filenames:
    num = re.findall(r"\d+\.?\d*",i)
    precent = num[0]
    data_pca = pd.read_csv(i,header=None)
    data_pca.rename(columns={data_pca.columns[-1]:'label'}, inplace=True)
    X = data_pca.iloc[:180,:-1]
    y = data_pca.iloc[:180,-1]
    y = y.apply(lambda x: 0 if x==0 else 1)
    clf = RandomForestClassifier(max_depth=2)
    scores = cvs( X, y,clf)
    print('Current PCA precentage is: ', precent)
    print('The mean accuracy score is: ', np.mean(scores))
    print('-'*50)
    acc.append(np.mean(scores))

# Plot accuracy of different PCA precentages
plt.plot([61,70,80,85,90,100],acc)
plt.xlabel('PCA Precentage')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different PCA Precentages')
plt.show()

# MULTIPLE CLASSIFICATION

# Multiple classification using KNN
filenames = ['clf_pca61%.csv','clf_pca70%.csv','clf_pca80%.csv','clf_pca85%.csv','clf_pca90%.csv','clf_pca100%.csv']
acc = []
for i in filenames:
    num = re.findall(r"\d+\.?\d*",i)
    precent = num[0]
    data_pca = pd.read_csv(i,header=None)
    data_pca.rename(columns={data_pca.columns[-1]:'label'}, inplace=True)
    X = data_pca.iloc[:90,:-1]
    y = data_pca.iloc[:90,-1]
    clf = KNeighborsClassifier(n_neighbors= 10)
    scores = cvs( X, y,clf)
    print('Current PCA precentage is: ', precent)
    print('The mean accuracy score is: ', np.mean(scores))
    print('-'*50)
    acc.append(np.mean(scores))

# Plot accuracy of different PCA precentages
plt.plot([61,70,80,85,90,100],acc)
plt.xlabel('PCA Precentage')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different PCA Precentages')
plt.show()

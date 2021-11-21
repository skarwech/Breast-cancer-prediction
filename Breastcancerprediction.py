import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from warnings import filterwarnings
filterwarnings("ignore")

#importing data
cancer_data = pd.read_csv("C:/Users/GCBµ/Downloads/Compressed/archive/data.csv")
cancer_data = cancer_data.drop('id', axis=1)
cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.contains('^Unnamed')]


#getting the data infos
cancer_data.head()
cancer_data.info()
cancer_data.describe(include='all')

#encoding the data
cancer_data.diagnosis = cancer_data.diagnosis.astype('category')
cancer_data.diagnosis = cancer_data.diagnosis.cat.codes
cancer_data.diagnosis.value_counts()
cancer_mean = cancer_data.loc[:, 'radius_mean':'fractal_dimension_mean']
cancer_mean['diagnosis'] = cancer_data['diagnosis']


#Scatterplot matrix
dimensions = []
for col in cancer_mean:
    dimensions.append(dict(label = col, values = cancer_mean[col]))
fig = go.Figure(data = go.Splom(dimensions = dimensions[:-2],showupperhalf=False, diagonal_visible=False,marker=dict(color='rgba(135, 206, 250, 0.5)',size=5,line=dict(color='MediumPurple',width=0.5))))
fig.update_layout(title='Pairplot for mean attributes of the dataset',width=1100,height=1500,)
fig.show()

#Correlation matrix
plt.figure(figsize = (40, 24), dpi = 70)
corr = cancer_data.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr,mask = mask,cmap = 'Blues',annot = True,linewidths = 0.3,fmt = ".2f")
plt.title('Correlation Matrix',fontsize = 20,weight = 'bold',color = '#1581c5')
plt.show()

def subplot_titles(cols):
    '''
    Creates titles for the subplot's subplot_titles parameter.
    '''
    titles = []
    for i in cols:
        titles.append(i+' : Distribution')
        titles.append(i+' : Violin plot')
        titles.append(i+' by Diagnosis')
    
    return titles
def subplot(cols, row = 0, col = 3):
    '''
    Takes a dataframe as an input and returns distribution plots for each variable.
    '''
    row = len(cols)
    fig = make_subplots(rows=row, cols=3, subplot_titles = subplot_titles(cols))
    for i in range(row):
        fig.add_trace(go.Histogram(x = cancer_data[ cols[i] ],opacity = 0.7),row=i+1, col=1)
        fig.add_trace(go.Violin(y = cancer_data[cols[i]],box_visible=True),row=i+1, col=2)
        fig.add_trace(go.Box(y = cancer_data[ cols[i] ][cancer_data.diagnosis == 0],marker_color = '#6ce366',name = 'Benign'),row=i+1, col=3)
        fig.add_trace(go.Box(y = cancer_data[ cols[i] ][cancer_data.diagnosis == 1],marker_color = '#de5147',name = 'Malignant'),row=i+1, col=3)
    for i in range(row):
        fig.update_xaxes(title_text = cols[i], row=i+1)
    fig.update_yaxes(title_text="Count")
    fig.update_layout(height= 450*row, width=1100,title = 'Summary of mean tumor attributes (For Diagnois : Green=Benign, Red=Malignant)',showlegend = False,plot_bgcolor="#f7f1cb")    
    fig.show()
x = subplot(cancer_data.drop('diagnosis', axis=1).columns)

#data processing
def outlier(df):
    df_ = df.copy()
    df = df.drop('diagnosis', axis=1)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 -(1.5 * iqr) 
    upper_limit = q3 +(1.5 * iqr)
    for col in df.columns:
        for i in range(0,len(df[col])):
            if df[col][i] < lower_limit[col]:            
                df[col][i] = lower_limit[col]
            if df[col][i] > upper_limit[col]:            
                df[col][i] = upper_limit[col]    
    for col in df.columns:
        df_[col] = df[col]
    return(df_)

cancer_data = outlier(cancer_data)
X = cancer_data.drop('diagnosis', axis=1)
y = cancer_data.diagnosis

#studying multicollinearity
def VIF(df):
    vif = pd.DataFrame()
    vif['Predictor'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, col) for col in range(len(df.columns))]
    return vif

vif_df = VIF(X).sort_values('VIF', ascending = False, ignore_index = True)
print(vif_df.head(8))

# Removing features with VIF > 10,000
high_vif_features = list(vif_df.Predictor.iloc[:2])
vif_features = X.drop(high_vif_features, axis=1)
print(vif_features)

#creatin test and train data
X_train, X_test, y_train, y_test = train_test_split(vif_features, y, test_size = 0.2, random_state = 39)

# Logistic regression with VIF features , BaggingClassifier and hyperparameter tuning
steps = [('scaler', StandardScaler()), ('log_reg', LogisticRegression())]
pipeline = Pipeline(steps)
parameters = dict(log_reg__solver = ['newton-cg', 'lbfgs', 'liblinear'],log_reg__penalty =  ['l2'],log_reg__C = [100, 10, 1.0, 0.1, 0.01])
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1,error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
log_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(log_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.title('Confusion Matrix for Logistic Regression',fontsize = 15,weight = 'semibold',color = '#1581c5')
plt.show()

#KNN with VIF features and hyperparameter tuning
steps = [('scaler', StandardScaler()),('knn', BaggingClassifier(KNeighborsClassifier()))]
pipeline = Pipeline(steps)
parameters = dict(knn__base_estimator__metric = ['euclidean', 'manhattan', 'minkowski'],knn__base_estimator__weights =  ['uniform', 'distance'],knn__base_estimator__n_neighbors = range(2,15),knn__bootstrap = [True, False],knn__bootstrap_features = [True, False],knn__n_estimators = [5])
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1,)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
knn_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(knn_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.title('Confusion Matrix for K-Nearest Neighbor',fontsize = 15,weight = 'semibold',color = '#1581c5')
plt.show()

# SVC with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),('svc', SVC())]
pipeline = Pipeline(steps)
parameters = dict(svc__kernel = ['poly', 'rbf', 'sigmoid'],svc__gamma =  [0.0001, 0.001, 0.01, 0.1],svc__C = [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20])
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
svc_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(svc_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
plt.title('Confusion Matrix for Support Vector Machine',fontsize = 15,weight = 'semibold',color = '#1581c5')
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.show()

# Random Forest Classifier with VIF features and hyperparameter tuning
steps = [('scaler', StandardScaler()),('rf', RandomForestClassifier(random_state = 0))]
pipeline = Pipeline(steps)
parameters = dict(rf__n_estimators = [10,100], rf__max_features = ['sqrt', 'log2'],)
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
rf_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(rf_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.title('Confusion Matrix for Random Forest',fontsize = 15,weight = 'semibold',color = '#1581c5')
plt.show()

# Ridge Classifier with VIF features and hyperparameter tuning
steps = [('scaler', StandardScaler()),('ridge', RidgeClassifier())]
pipeline = Pipeline(steps)
parameters = dict(ridge__alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1,error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
ridge_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(ridge_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
plt.title('Confusion Matrix for Ridge Classifier',fontsize = 15,weight = 'semibold',color = '#1581c5')
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.show()

# Gradient Boosting Classifier  with VIF features  and hyperparameter tuning
steps = [('scaler', StandardScaler()),('gbc', GradientBoostingClassifier())]
pipeline = Pipeline(steps)
parameters = dict(gbc__n_estimators = [10,100,200],gbc__loss = ['deviance', 'exponential'],gbc__learning_rate = [0.001, 0.1, 1, 10])
cv = GridSearchCV(pipeline,param_grid = parameters,cv = 5,scoring = 'accuracy',n_jobs = -1,error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
gb_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(gb_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
plt.title('Confusion Matrix for Gradient Boosting',fontsize = 15,weight = 'semibold',color = '#1581c5')
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.show()

xgb = XGBClassifier(max_depth = 5,min_child_weight = 1,gamma = 0.3,subsample = 0.8,colsample_bytree = 0.8,learning_rate = 0.1,reg_alpha=0.05,disable_default_eval_metric = True)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_pred, y_test) * 100

print('Best parameters :\n{}\n'.format(cv.best_params_))
print('Accuracy :\n{}\n'.format(xgb_accuracy))
print('Classification report :\n{}\n'.format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_pred, y_test)
plt.title('Confusion Matrix for XGBoost',fontsize = 15,weight = 'semibold',color = '#1581c5')
sns.heatmap(cm, cmap = 'Blues',annot = True, fmt='d')
plt.show()

# Accuracies of models
results = {'Model' :['Logistic Regression', 'KNN', 'SVC', 'Random Forest', 'Rigde Classifier', 'Gradient Boosting', 'XGBoost'],'Accuracy' : [log_accuracy, knn_accuracy, svc_accuracy, rf_accuracy, ridge_accuracy, gb_accuracy, xgb_accuracy]}
results = pd.DataFrame(results).sort_values('Accuracy', ignore_index=True, ascending=False)
results.Accuracy = results.Accuracy.round(2)
results

fig = px.line(results,x = results.Model,y = results.Accuracy,text=results.Accuracy,)
fig.update_traces(textposition = 'top right')
fig.update_layout(title = 'Model vs Accuracy',plot_bgcolor = '#f9faed')
fig.show()